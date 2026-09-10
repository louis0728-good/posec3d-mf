# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, Dict, List, Optional, Union
import numpy as np

import mmengine
from mmengine.logging import MMLogger
from .pose_dataset import PoseDataset
from mmaction.registry import DATASETS
from .base import BaseActionDataset
import warnings

"""
pkl 裡應該要有 label_upper、label_lower、keypoint、keypoint_score、total_frames、img_shape，以及 split
輸出 PKL 結構：
{
    'split': {
        'xsub_train': ['test001_p1', 'test002_p2', ...],
        'xsub_val':   ['test001_p2', 'test003_p1', ...] # 影片名稱
    },
    'annotations': [
        {
            'frame_dir': 'test001_p1',
            'label_upper': 0,
            'label_lower': 3,
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 150,
            'keypoint': np.array(...),       # (1, T, 17, 2)
            'keypoint_score': np.array(...)  # (1, T, 17)
        },
        ...
    ]
}
"""
@DATASETS.register_module() # Registry 註冊裝飾器
# 是把 class 物件放進某個 registry 的字典裡：
class DualPoseDataset(PoseDataset):
    def __init__(self, *args, window_len=21, window_stride=1, max_windows=None, **kwargs):
        self.window_len = window_len
        self.window_stride = window_stride
        self.max_windows = max_windows
        super().__init__(*args, **kwargs)      # 必須放在最後一行

    def load_data_list(self):
        data_list = super().load_data_list()   # 先讓 PoseDataset 讀 pkl + 依 split 過濾
        L, S = self.window_len, self.window_stride
        out, n_pad = [], 0

        for item in data_list:
            T = int(item['total_frames'])

            if T < L:                          # 不足：複製最後一幀補滿，絕不 wrap
                pad = L - T
                new_item = dict(item)
                for k in ('keypoint', 'keypoint_score'):
                    if k in item:
                        a = item[k]
                        new_item[k] = np.concatenate(
                            [a, np.repeat(a[:, -1:], pad, axis=1)], axis=1)
                new_item['total_frames'] = L
                new_item['clip_start'] = 0
                out.append(new_item)
                n_pad += 1
                continue

            starts = list(range(0, T - L + 1, S))
            if self.max_windows is not None and len(starts) > self.max_windows:
                # 均勻取樣，仍涵蓋整段 clip，只是變稀疏
                starts = sorted(set(
                    np.linspace(0, T - L, self.max_windows).round().astype(int).tolist()))

            for s in starts:   # 滑動：0, S, 2S, ... 直到切完
                new_item = dict(item)          # 淺拷貝，底下才換掉 keypoint
                for k in ('keypoint', 'keypoint_score'):
                    if k in item:
                        new_item[k] = item[k][:, s:s + L]
                new_item['total_frames'] = L
                new_item['clip_start'] = s
                out.append(new_item)

        MMLogger.get_current_instance().info(
            f'[DualPoseDataset] split={self.split}: '
            f'{len(data_list)} 筆 → {len(out)} 個視窗 '
            f'(len={L}, stride={S}, padded={n_pad})')
        return out

    def get_data_info(self, idx: int) -> Dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx) # 這裡應該是代表開始看我的 pkl 
                
        if 'label_upper' in data_info and 'label_lower' in data_info:
            label_upper = int(data_info['label_upper'])
            label_lower = int(data_info['label_lower'])

            if label_upper < 0 or label_lower < 0:
                raise ValueError(
                    f'[DualPoseDataset] 第 {idx} 筆資料 label 有負值: '
                    f'label_upper={label_upper}, label_lower={label_lower}'
                )
            if label_upper >= 2 or label_lower >= 6: # 上半身 2 個動作，下半身 7 個
                raise ValueError(
                    f'[DualPoseDataset] 第 {idx} 筆資料 label 怪怪的: '
                    f'label_upper={label_upper}, label_lower={label_lower}'
                )
            data_info['label'] = [label_upper, label_lower]
            
        else:
            raise KeyError(
                f'[DualPoseDataset] 第 {idx} 筆資料缺少 label_upper 或 label_lower，'
                f'現有 keys: {[k for k in data_info.keys() if "label" in k.lower()]}') # 輸出 list 有那些東西(frame_dir, ...)?
        """ result:
            'frame_dir': 'test001_p1',
            'label: [label_upper, label_lower]
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 150,
            'keypoint': np.array(...),       # (1, T, 17, 2)
            'keypoint_score': np.array(...)  # (1, T, 17)
        """
        return data_info # data_info（也就是後面 pipeline 裡的 results）是單筆資料：

