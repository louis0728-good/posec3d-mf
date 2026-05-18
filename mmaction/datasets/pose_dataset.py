# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, Dict, List, Optional, Union

import mmengine
from mmengine.logging import MMLogger

from mmaction.registry import DATASETS
from .base import BaseActionDataset


@DATASETS.register_module()
class PoseDataset(BaseActionDataset):
    """
    這個資料集會載入人體姿態資料，並套用指定的 transforms，
    最後回傳一個包含姿態資訊的 dict。

    ann_file 是一個 pickle 檔案，而其中的 json 檔內容是一串標註資料清單。
    每一筆標註資料包含的欄位有：
    frame_dir（影片 id）、total_frames、label、kp、kpscore。

    參數：
        ann_file (str)：標註檔案的路徑。
        
        pipeline (list[dict | callable])：
            一系列資料轉換流程。

        split (str, optional)：
            指定使用哪個資料切分。
            對於 UCF101 和 HMDB51，可選：
            'train1', 'test1', 'train2', 'test2', 'train3', 'test3'
            
            對於 NTURGB+D，可選：
            'xsub_train', 'xsub_val', 'xview_train', 'xview_val'
            
            對於 NTURGB+D 120，可選：
            'xsub_train', 'xsub_val', 'xset_train', 'xset_val'
            
            對於 FineGYM，可選：
            'train', 'val'
            
            預設為 None。

        valid_ratio (float, optional)：
            給 KineticsPose 用的有效比例。
            假設一支影片有 n 幀，只有在其中有人體姿態的幀數
            至少達到 n * valid_ratio 時，這支影片才算是有效的訓練樣本。
            如果是 None，表示不使用這個條件。
            （只適用於 Kinetics Pose）
            預設為 None。

        box_thr (float)：
            人體偵測框的信心分數門檻。
            只有 confidence score 大於 `box_thr` 的框才會被保留。
            如果是 None，表示不使用這個條件。
            （只適用於 Kinetics）
            可選值為 0.5、0.6、0.7、0.8、0.9。
            預設為 0.5。
    """

    def __init__(self,
                 ann_file: str, # The ann_file is a pickle file
                 pipeline: List[Union[Dict, Callable]],
                 split: Optional[str] = None,
                 valid_ratio: Optional[float] = None,
                 box_thr: float = 0.5,
                 **kwargs) -> None:
        self.split = split
        self.box_thr = box_thr
        assert box_thr in [.5, .6, .7, .8, .9]
        self.valid_ratio = valid_ratio

        super().__init__(
            ann_file, pipeline=pipeline, modality='Pose', **kwargs)

    def load_data_list(self) -> List[Dict]:
        """Load annotation file to get skeleton information.""" # 讀 pkl
        assert self.ann_file.endswith('.pkl')
        mmengine.exists(self.ann_file)
        data_list = mmengine.load(self.ann_file) # 這時候還有全部的東西

        if self.split is not None:
            split, annos = data_list['split'], data_list['annotations']
            identifier = 'filename' if 'filename' in annos[0] else 'frame_dir' #　identifier = frame_dir
            """ annotations': [
                        {
                            'frame_dir....,
                            ...
                        }
                        這是我的 anno
            """
            split = set(split[self.split]) # split = set(data_list['split']['xsub_train']) 就是這個意思
            # 結果會變成像這樣： {'test001_p1', 'test002_p2', ...} 有點像是影片的目錄，可以知道影片的 key 

            data_list = [x for x in annos if x[identifier] in split] #　這裡應該是把　frmae_dir 記錄起來 
            # 如果 這個 annon 裡的 x[frame_dir] 有在 xsub_train 裡面那就記錄起來這個 key 
            # data_list 現在剩 'annotations': 裡的資料

        # Sometimes we may need to load video from the file'
        # 應該用不到
        if 'video' in self.data_prefix:
            for item in data_list:
                if 'filename' in item:
                    item['filename'] = osp.join(self.data_prefix['video'],
                                                item['filename'])
                if 'frame_dir' in item:
                    item['frame_dir'] = osp.join(self.data_prefix['video'],
                                                 item['frame_dir'])
        return data_list

    def filter_data(self) -> List[Dict]:
        """Filter out invalid samples."""
        if self.valid_ratio is not None and isinstance(
                self.valid_ratio, float) and self.valid_ratio > 0:
            self.data_list = [
                x for x in self.data_list if x['valid'][self.box_thr] /
                x['total_frames'] >= self.valid_ratio
            ]
            for item in self.data_list:
                assert 'box_score' in item,\
                    'if valid_ratio is a positive number,' \
                    'item should have field `box_score`'
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds

        logger = MMLogger.get_current_instance()
        logger.info(
            f'{len(self.data_list)} videos remain after valid thresholding')

        return self.data_list

    def get_data_info(self, idx: int) -> Dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx) # 原本這裡應該會取用 包括 label 之內的東西，但我有上下半身標籤，所以我自建一個新的

        # Sometimes we may need to load skeleton from the file
        if 'skeleton' in self.data_prefix:
            identifier = 'filename' if 'filename' in data_info \
                else 'frame_dir'
            ske_name = data_info[identifier] # [frame_dir]
            ske_path = osp.join(self.data_prefix['skeleton'],
                                ske_name + '.pkl')
            ske = mmengine.load(ske_path)
            for k in ske:
                data_info[k] = ske[k]

        return data_info
