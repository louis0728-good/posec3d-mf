# Copyright (c) OpenMMLab. All rights reserved.
import torch
import logging
logger = logging.getLogger(__name__)
from mmaction.registry import MODELS
from mmaction.models.recognizers import Recognizer3D

@MODELS.register_module()
class CustomDualRecognizer(Recognizer3D):
    """3D recognizer model framework."""

    """
    完全基於官方架構，只修復測試模式下沒有將 data_samples 傳給 neck 的問題。
    讓 Neck 可以安全地從 data_samples 中挖出 sg_features。
    """
    def extract_feat(self,
                     inputs,
                     stage='neck',
                     data_samples = None,
                     test_mode= False):

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()
        logger.info("inputs shape=%s", tuple(inputs.shape)) # 應該會是 (B, 1, 17, 21, 56, 56)
        logger.info("inputs C (shape[2])=%s", inputs.shape[2])
        """ inputs = batch(packed_results)
            packed_results['inputs']       → 一筆 heatmap (1, 17, 21, 56, 56)
            packed_results['data_samples'] → 一個 ActionDataSample
            data_sample = gt_label、sg_features ...
            gt_label = (results['label'])
        """
        num_segs = inputs.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W] 其實應該就可以叫 [B, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1, ) + inputs.shape[2:])
        logger.debug("N 與 num_crops 合併後 shape: %s", tuple(inputs.shape)) # 

        # Check settings of test
        if test_mode:
            if self.test_cfg is not None:
                loss_predict_kwargs['fcn_test'] = self.test_cfg.get(
                    'fcn_test', False)
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)

                total_views = inputs.shape[0] # 跟那個 CLIP/CROP 有關
                assert num_segs == total_views, (
                    'max_testing_views is only compatible '
                    'with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views]
                    feat = self.backbone(batch_imgs)
                    if self.with_neck:
                        # 補上 data_samples，因為 neck（DualWindowGatingNeck）需要從 data_samples 裡取出 sg_features
                        feat, _ = self.neck(feat, data_samples=data_samples) #　跳進了 dual_window_gating.py
                        # self.neck 其實不是一個普通的函數 (def)，而是一個 DualWindowGatingNeck 的物件 (Object)。
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
                x = self.backbone(inputs)
                if self.with_neck:
                    # 補上 data_samples
                    x, _ = self.neck(x, data_samples=data_samples) # 跳進了 dual_window_gating.py
                    # 原作者有寫但因為他沒有定義 neck，現在我的 config 有。可能是這個原因所以我的 neck 會有用。 

            return x, loss_predict_kwargs
        else:
            # Return features extracted through backbone
            x = self.backbone(inputs)
            if stage == 'backbone':
                return x, loss_predict_kwargs

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
            
    def predict(self, inputs, data_samples, **kwargs): # 覆寫
        feats, predict_kwargs = self.extract_feat(
            inputs, data_samples=data_samples, test_mode=True)
        predictions = self.cls_head.predict(
            feats, data_samples, **predict_kwargs)
        return predictions
