# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer


@MODELS.register_module()
class Recognizer3D(BaseRecognizer): # Recognizer3D 是 BaseRecognizer 的子類別。Recognizer3D 會「繼承」BaseRecognizer 的方法與屬性
    # BaseRecognizer 在 同路徑的 base.py
    """3D recognizer model framework."""
    # 根據 train/test 模式，抽取 backbone / neck / head 不同階段的特徵
    def extract_feat(self,
                     inputs: Tensor, #　通常形狀是　[N, crops, C, T, H, W]
                     stage: str = 'neck',
                     data_samples: OptSampleList = None, # It usually includes information such as ``gt_label`
                     test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (torch.Tensor): The input data.
            stage (str): Which stage to output the feature. 輸出特徵圖到 neck
                Defaults to ``'neck'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                torch.Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``loss_aux``.
        """

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()
        # 這是一個暫存字典，用來把中途產生的額外資訊，傳給後面的 loss() 或 predict()

        num_segs = inputs.shape[1] # 取出 crop 的數量。
        # crop：同一段時間再做不同空間裁切（通常是不同位置的空間裁切）
        # ok 確定 input.shape 會長 [N, num_crops, C, T, H, W] 

        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        # num_crops 應該就是 config 寫的 num_clips ; clip_len=21：控制 T。
        # 所以以官方的設定來說應該會是 inputs shape: [N, 10, C, 48, H, W]。
        inputs = inputs.view((-1, ) + inputs.shape[2:]) 
        # view 會把資料 shape -1 就是讓程式自己去推倒，反正這裡的 n 會 * num_crops
        """ [N, num_crops, C, T, H, W] -> [N * num_crops, C, T, H, W] """
        # 所以 recognizer 先把多 clips 展平，讓 backbone 把每個 clip 當成一筆獨立資料算特徵。

        # Check settings of test
        # test_cfg (Union[ConfigDict, dict], optional): Config for testing.Defaults to None.
        if test_mode:
            if self.test_cfg is not None:
                # test_cfg = dict(type='TestLoop')
                loss_predict_kwargs['fcn_test'] = self.test_cfg.get(
                    'fcn_test', False)
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False): # 如果 True 啟用「分批跑 views」的模式。
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)

                total_views = inputs.shape[0]
                assert num_segs == total_views, (
                    'max_testing_views is only compatible '
                    'with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views] # 取出一小段 views 當一個 mini-batch
                    feat = self.backbone(batch_imgs)
                    if self.with_neck: # with_neck 會 回傳 true or false
                        """
                        with_neck == True ⟺ self.neck 這個屬性存在，而且不是 None
                        with_neck == False ⟺ 沒有 self.neck，或 self.neck is None
                        """
                        feat, _ = self.neck(feat) # (處理後的特徵, 額外資訊)
                    feats.append(feat)
                    view_ptr += max_testing_views

                def recursively_cat(feats):
                    # recursively traverse feats until it's a tensor,
                    # then concat
                    out_feats = []
                    for e_idx, elem in enumerate(feats[0]):
                        batch_elem = [feat[e_idx] for feat in feats]
                        if not isinstance(elem, torch.Tensor):
                            batch_elem = recursively_cat(batch_elem)
                        else:
                            batch_elem = torch.cat(batch_elem)
                        out_feats.append(batch_elem)

                    return tuple(out_feats)

                if isinstance(feats[0], tuple):
                    x = recursively_cat(feats)
                else:
                    x = torch.cat(feats)
            else:
                x = self.backbone(inputs) # self.backbone_from = 'mmaction2'
                # 我的 backbone type 應該是 ResNet3dSlowOnly
                if self.with_neck:
                    x, _ = self.neck(x) # (處理後的特徵, 額外資訊)
                # neck (Union[ConfigDict, dict], optional): Neck for feature fusion. Defaults to None.
                # neck type 是 DualWindowGatingNeck
                
            return x, loss_predict_kwargs
        else:
            # Return features extracted through backbone
            x = self.backbone(inputs)
            if stage == 'backbone':
                return x, loss_predict_kwargs # x 很有可能就是特徵圖

            loss_aux = dict()
            if self.with_neck:
                x, loss_aux = self.neck(x, data_samples=data_samples)

            # Return features extracted through neck
            loss_predict_kwargs['loss_aux'] = loss_aux
            if stage == 'neck':
                return x, loss_predict_kwargs

            # Return raw logits through head.
            if self.with_cls_head and stage == 'head':
                x = self.cls_head(x, **loss_predict_kwargs)
                return x, loss_predict_kwargs
