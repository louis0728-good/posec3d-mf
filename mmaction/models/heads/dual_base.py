# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, SampleList


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Defaults to 1.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class DualBaseHead(BaseModule, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - :meth:`forward`, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Defaults to False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Defaults to 0.
        topk (int or tuple): Top-k accuracy. Defaults to ``(1, 5)``.
        average_clips (dict, optional): Config for averaging class
            scores over multiple clips. Defaults to None.
        init_cfg (dict, optional): Config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                num_classes_upper: int,
                num_classes_lower: int,
                in_channels: int,
                loss_cls: Dict = dict(
                    type='CrossEntropyLoss', loss_weight=1.0),
                loss_cls_upper: Optional[Dict] = None,   # 新增 下半身的 loss config
                loss_cls_lower: Optional[Dict] = None,   # 新增 上半身的 loss config
                loss_weight_upper: float = 1.0,          # 新增 上半身的懲罰權重
                loss_weight_lower: float = 1.0,          # 新增 下半身的懲罰權重
                multi_class: bool = False,
                label_smooth_eps: float = 0.0,
                topk: Union[int, Tuple[int]] = (1, ),
                average_clips: Optional[Dict] = None,
                debug: bool = False,              # ← 新增
                init_cfg: Optional[Dict] = None) -> None:
        super(DualBaseHead, self).__init__(init_cfg=init_cfg)
        self.num_classes_upper = num_classes_upper
        self.num_classes_lower = num_classes_lower
        self.in_channels = in_channels
        # 兩個頭各建一份 loss，才能對 2 類 / 6 類分別套用 class_weight
        self.loss_cls_upper = MODELS.build(loss_cls_upper or loss_cls)
        self.loss_cls_lower = MODELS.build(loss_cls_lower or loss_cls)
        self.loss_weight_upper = loss_weight_upper
        self.loss_weight_lower = loss_weight_lower

        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.average_clips = average_clips
        self.debug = debug
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

    @abstractmethod
    def forward(self, x, **kwargs) -> ForwardResults:
        """Defines the computation performed at every call."""
        raise NotImplementedError

    def loss(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
             data_samples: SampleList, **kwargs) -> Dict:
        # 呼叫 forward 取得分類分數，再呼叫 loss_by_feat 計算損失。是訓練時的入口。
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)
        """
        核心損失計算邏輯：
            從 data_samples 取出 ground truth 標籤並堆疊。
            處理邊界情況（標量標籤、batch size=1 的 soft label）。
            如果是硬標籤，計算 top-k accuracy 作為指標。
            若啟用 label smoothing，將硬標籤轉為 one-hot 後做平滑。
        """

    def loss_by_feat(self, cls_scores: Tuple[torch.Tensor, torch.Tensor],
                     data_samples: SampleList) -> Dict:
        """專為雙頭輸出的 Loss 計算"""

        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        # 解開 Tuple 得到上下半身的分數
        cls_score_upper, cls_score_lower = cls_scores
        """
        data_sample = gt_label、sg_features ...
                gt_label = (results['label'])
        """
        # 處理 Ground Truth 標籤 (我們在 Dataset 包成了 [upper, lower])
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_score_upper.device) # 則是確保答案跟考卷放在同一個 GPU 記憶體裡面，不然程式會當機。
        labels = labels.squeeze()
        """
        sample[0] gt_label shape=(2,)
        sample[1] gt_label shape=(2,)
        ...
        after stack/squeeze labels shape=(16, 2) 
        如果想這樣那就是正常。

        sample[0] gt_label shape=(1, 2, 1) 不正常
        """

        if labels.dim() == 1:
            labels = labels.view(-1, 2)  # 確保維持 [B, 2]
        assert labels.dim() == 2 and labels.shape[1] == 2, \
            f"[DualBaseHead] labels shape 異常: {labels.shape}, 預期 [B, 2]"

        labels_upper = labels[:, 0] # labels[:, 0]: 我要「所有橫列 (:)」的「第 0 個直欄 (0)」，也就是把上半身的答案全部抽出來。
        labels_lower = labels[:, 1] # abels[:, 1] 則是抽出「第 1 個直欄」，也就是下半身答案。

        losses = dict()
        if self.debug:
            if labels.min().item() < 0:
                raise ValueError(f'[DualBaseHead] 發現負標籤: labels={labels}')

            if labels_upper.max().item() >= cls_score_upper.shape[1]:
                raise ValueError(
                    f'[DualBaseHead] upper label 越界: max={labels_upper.max().item()}, '
                    f'num_classes={cls_score_upper.shape[1]}'
                )

            if labels_lower.max().item() >= cls_score_lower.shape[1]:
                raise ValueError(
                    f'[DualBaseHead] lower label 越界: max={labels_lower.max().item()}, '
                    f'num_classes={cls_score_lower.shape[1]}'
                )

            if cls_score_upper.shape[0] != labels.shape[0] or cls_score_lower.shape[0] != labels.shape[0]:
                raise ValueError(
                    f'[DualBaseHead] batch size 對不上: '
                    f'upper={cls_score_upper.shape[0]}, lower={cls_score_lower.shape[0]}, labels={labels.shape[0]}'
                )

            losses = dict()

            assert labels_upper.max() < cls_score_upper.shape[1], \
                f"[DualBaseHead] upper label {labels_upper.max()} >= num_classes {cls_score_upper.shape[1]}"
            assert labels_lower.max() < cls_score_lower.shape[1], \
                f"[DualBaseHead] lower label {labels_lower.max()} >= num_classes {cls_score_lower.shape[1]}"
    
        # top-1 準確率：全程留在 GPU，避免每 iter 4 次強制同步
        with torch.no_grad():
            losses['top1_acc_upper'] = (
                cls_score_upper.argmax(1) == labels_upper).float().mean()
            losses['top1_acc_lower'] = (
                cls_score_lower.argmax(1) == labels_lower).float().mean()
                
        # === label smoothing（原本收了參數卻沒實作）===
        tgt_upper, tgt_lower = labels_upper, labels_lower
        if self.label_smooth_eps != 0:
            e = self.label_smooth_eps
            nu, nl = cls_score_upper.shape[1], cls_score_lower.shape[1]
            tgt_upper = (1 - e) * F.one_hot(labels_upper, nu).float() + e / nu
            tgt_lower = (1 - e) * F.one_hot(labels_lower, nl).float() + e / nl

        losses['loss_cls_upper'] = self.loss_weight_upper * \
            self.loss_cls_upper(cls_score_upper, tgt_upper)
        losses['loss_cls_lower'] = self.loss_weight_lower * \
            self.loss_cls_lower(cls_score_lower, tgt_lower)

        return losses

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
    # 推論入口：呼叫 forward 取得分數，再交給 predict_by_feat 產生預測結果。
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: Tuple[torch.Tensor, torch.Tensor],
                        data_samples: SampleList) -> SampleList:
        """
            根據片段數（num_segs）呼叫 average_clip 平均多片段分數。
            取 argmax 得到預測標籤。
            將預測分數和標籤寫回每個 data_sample。
        """
        """專為雙頭輸出的預測結果打包"""
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        """
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label in zip(data_samples, cls_scores,
                                                  pred_labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label)
        """

        cls_score_upper, cls_score_lower = cls_scores
        if self.debug:  
            if not torch.isfinite(cls_score_upper).all():
                raise FloatingPointError('[DualBaseHead] cls_score_upper 出現 NaN 或 Inf')

            if not torch.isfinite(cls_score_lower).all():
                raise FloatingPointError('[DualBaseHead] cls_score_lower 出現 NaN 或 Inf')

        num_segs = cls_score_upper.shape[0] // len(data_samples)

        # 取片段平均
        cls_score_upper = self.average_clip(cls_score_upper, num_segs=num_segs)
        cls_score_lower = self.average_clip(cls_score_lower, num_segs=num_segs)

        # 取最大值做預測標籤
        pred_labels_upper = cls_score_upper.argmax(dim=-1, keepdim=True).detach()
        pred_labels_lower = cls_score_lower.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score_u, score_l, pred_u, pred_l in zip(
                data_samples, cls_score_upper, cls_score_lower,
                pred_labels_upper, pred_labels_lower):
            
            # 存入自定義的上下半身欄位
            # 原本的 data_sample 沒有上下半身的格子，
            data_sample.set_field(score_u, 'pred_score_upper')
            data_sample.set_field(pred_u, 'pred_label_upper')
            data_sample.set_field(score_l, 'pred_score_lower')
            data_sample.set_field(pred_l, 'pred_label_lower')
            
            # 給底層一個預設值防呆
            #    DualAccMetric 讀的是上面的 pred_score_upper / pred_score_lower。
            #    若改用原生 AccMetric、或 tools/test.py --dump，
            #    拿到的會「只有上半身」，且 num_classes metainfo 會被寫成 2。
            data_sample.set_pred_score(score_u) 
            data_sample.set_pred_label(pred_u)

        return data_samples

    def average_clip(self,
                     cls_scores: torch.Tensor,
                     num_segs: int = 1) -> torch.Tensor:
        """Averaging class scores over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_scores (torch.Tensor): Class scores to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class scores.
        """

        if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view((batch_size // num_segs, num_segs) +
                                     cls_scores.shape[1:])

        if self.average_clips is None:
            return cls_scores
        elif self.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores
