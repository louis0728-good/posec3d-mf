# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
import torch
from torch import Tensor, nn
from mmaction.registry import MODELS
from mmaction.utils import ConfigType
#from .base import BaseHead
from .dual_base import DualBaseHead

@MODELS.register_module()
class DualI3DHead(DualBaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes_upper: int, # 上半身類別數
                 num_classes_lower: int, # 下半身
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes_upper=num_classes_upper, num_classes_lower=num_classes_lower, 
                         in_channels=in_channels, loss_cls=loss_cls, **kwargs)

        self.num_classes_upper = num_classes_upper
        self.num_classes_lower = num_classes_lower

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        
        self.fc_cls_upper = nn.Linear(self.in_channels, self.num_classes_upper)
        self.fc_cls_lower = nn.Linear(self.in_channels, self.num_classes_lower)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            # 兩層都要初始化
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls_upper, std=self.init_std)
        normal_init(self.fc_cls_lower, std=self.init_std)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if x.dim() != 5:
            raise ValueError(
                f'[DualI3DHead] 輸入特徵維度錯誤: got shape={tuple(x.shape)}, 預期 [B, C, T, H, W]'
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f'[DualI3DHead] 輸入通道數錯誤: got {x.shape[1]}, expected {self.in_channels}'
            )

        if not torch.isfinite(x).all():
            raise FloatingPointError('[DualI3DHead] 輸入特徵 x 出現 NaN 或 Inf')
        
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.reshape(x.shape[0], -1)

        assert x.shape[1] == self.in_channels, \
                f"[DualI3DHead] 特徵維度不匹配: expected {self.in_channels}, got {x.shape[1]}"

        # [N, in_channels]
        # 特徵分別通過兩個全連接層
        cls_score_upper = self.fc_cls_upper(x)
        cls_score_lower = self.fc_cls_lower(x)

        assert cls_score_upper.shape[1] == self.num_classes_upper, \
            f"[DualI3DHead] upper 輸出維度錯誤: {cls_score_upper.shape}"
        assert cls_score_lower.shape[1] == self.num_classes_lower, \
            f"[DualI3DHead] lower 輸出維度錯誤: {cls_score_lower.shape}"
        
        if not torch.isfinite(cls_score_upper).all():
            raise FloatingPointError('[DualI3DHead] cls_score_upper 出現 NaN 或 Inf')

        if not torch.isfinite(cls_score_lower).all():
            raise FloatingPointError('[DualI3DHead] cls_score_lower 出現 NaN 或 Inf')

        # [N, num_classes]
        return cls_score_upper, cls_score_lower # 同時回傳兩個預測結果
