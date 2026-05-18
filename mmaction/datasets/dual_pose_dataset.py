# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, Dict, List, Optional, Union

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
            if label_upper >= 2 or label_lower >= 7: # 上半身 2 個動作，下半身 7 個
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

