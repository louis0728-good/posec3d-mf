import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmengine.model import BaseModule

class MotionInject(nn.Module):
    """把 motion 特徵轉成 feat_dim 維，用來調變視窗特徵 F。

    beta : F + b
    gamma: F * (1 + g)
    film : F * (1 + g) + b

    末層 zero-init，初始輸出等同完全不注入。
    """

    def __init__(self, motion_dim, feat_dim, inject_type):
        super().__init__()
        assert inject_type in ('beta', 'gamma', 'film')
        self.inject_type = inject_type
        self.motion_norm = nn.BatchNorm1d(motion_dim)

        self.to_beta = None
        self.to_gamma = None
        if inject_type in ('beta', 'film'):
            self.to_beta = nn.Linear(motion_dim, feat_dim)
            nn.init.zeros_(self.to_beta.weight)
            nn.init.zeros_(self.to_beta.bias)
        if inject_type in ('gamma', 'film'):
            self.to_gamma = nn.Linear(motion_dim, feat_dim)
            nn.init.zeros_(self.to_gamma.weight)
            nn.init.zeros_(self.to_gamma.bias)

    def forward(self, feat_list, motion_list):
        """feat_list / motion_list 一一對應（長視窗、短視窗）。
        沿 batch 維 concat 過同一份 BN，讓長短視窗共用統計量。"""
        assert len(feat_list) == len(motion_list)
        n = len(feat_list)
        motion_all = self.motion_norm(torch.cat(motion_list, dim=0))

        beta_list = (self.to_beta(motion_all).chunk(n, dim=0)
                     if self.to_beta is not None else None)
        gamma_list = (self.to_gamma(motion_all).chunk(n, dim=0)
                      if self.to_gamma is not None else None)

        out = []
        for i, feat in enumerate(feat_list):
            if gamma_list is not None:
                feat = feat * (1.0 + gamma_list[i])
            if beta_list is not None:
                feat = feat + beta_list[i]
            out.append(feat)
        return out

    
@MODELS.register_module()
class DualWindowGatingNeck(BaseModule):
    def __init__(self, in_channels, motion_dim_whole=72, motion_dim_part=36,
                 out_channels=512, l2_in_channels=256, crop_margin=3, mode='per_gate',
                 motion_inject='none', inject_scope='whole_body', debug=False):
        super().__init__()
        self.in_channels = in_channels
        self.motion_dim_whole = motion_dim_whole
        self.out_channels = out_channels
        self.l2_in_channels = l2_in_channels
        self.crop_margin = crop_margin
        self.debug = debug
        self._printed_once = False
        self._step = 0
        assert mode in (
            'per_gate',
            'dual_gate',
            'dual_window',
            'learnable',
            'long_only',
            'short_only',
        ), f'[Neck] 不支援的 mode: {mode}'

        self.mode = mode

        assert motion_inject in ('none', 'beta', 'gamma', 'film'), \
            f'[Neck] 不支援的 motion_inject: {motion_inject}'
        assert inject_scope in ('whole_body', 'upper_lower'), \
            f'[Neck] 不支援的 inject_scope: {inject_scope}'
        
        self.motion_dim_part = motion_dim_part
        self.motion_inject = motion_inject
        self.inject_scope = inject_scope

        # 是否真的需要長、短視窗
        self.use_lw = mode != 'short_only'
        self.use_sw = mode != 'long_only'
        self.use_gate = mode in ('per_gate', 'dual_gate')                # 自適應 gating
        self.dual_gate = (mode == 'dual_gate')

        # 是否使用不依賴 motion 的可學融合
        self.use_learnable = mode == 'learnable'

        # 空間池化 (GAP)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # T 維保持一樣，後面的 h, w 變成 1
        # [B, C, T, H, W] -> [B, C, T, 1, 1]

        # 時間池化 (Temporal Pooling)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) # 跟上面差不多

        # layer2 是 256 通道，要投影到 512 才能跟 F_L 做逐通道 gating
        if self.use_sw:
            self.small_proj = nn.Sequential(
                nn.Linear(l2_in_channels, in_channels),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
            )

        """
            data_sample = gt_label、sg_features ...
            gt_label = (results['label'])
        """
        hidden_dim = out_channels // 2

        def _make_scorer(motion_dim):
            return nn.Sequential(
                nn.BatchNorm1d(motion_dim),
                nn.Linear(motion_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, in_channels),
            )

        if self.dual_gate:
            # 上下半身各一份、不共享權重：
            # 兩者的運動統計分布不同，且服務的是不同的分類頭
            self.motion_scorer_upper = _make_scorer(self.motion_dim_part)
            self.motion_scorer_lower = _make_scorer(self.motion_dim_part)
        elif self.use_gate:
            self.motion_scorer = _make_scorer(self.motion_dim_whole)

        if self.use_learnable:
            # 每個 channel 一個固定 α。
            # sigmoid(0) = 0.5，因此初始狀態等同 dual_window。
            self.alpha_logit = nn.Parameter(
                torch.zeros(in_channels)
            )

        if motion_inject != 'none':
            if inject_scope == 'upper_lower':
                # 上下半身各一份，權重不共享
                self.inject_upper = MotionInject(
                    self.motion_dim_part, in_channels, motion_inject)
                self.inject_lower = MotionInject(
                    self.motion_dim_part, in_channels, motion_inject)
            else:
                self.inject_whole = MotionInject(
                    self.motion_dim_whole, in_channels, motion_inject)

    def forward(self, x, data_samples=None, **kwargs): # 接收參數改為 data_samples
        if isinstance(x, (tuple, list)):
            assert len(x) == 2, f'[Neck] 預期 2 個 stage，收到 {len(x)}'
            x2, x3 = x 
            # x2 是 layer2，x3 是 layer3，所以時序來說，x2 的 T 會比 x3 長，因為 layer3 有做 temporal stride。
            # 但實際上 x3 的感受野更大，是代表最一開始那個 21 幀。x2 只是為了要等等的短視窗 S-G 做裁切。
        else:
            assert not self.use_sw, '[Neck] per_gate/dual_window 需要 out_indices=(1,2)'
            x2, x3 = None, x

        B = None
        if self.use_lw:
            assert x3 is not None, '[Neck] 此 mode 需要 layer3 特徵'
            B, C, T, H, W = x3.shape
            assert C == self.in_channels, f" layer 3 通道數不符: {C} vs {self.in_channels}"

        if self.use_sw:
            assert x2 is not None, '[Neck] 此 mode 需要 layer2 特徵'
            B_short = x2.shape[0]
            if B is None:
                B = B_short
            else:
                assert B == B_short, \
                    f'長短視窗 batch 不符: {B} vs {B_short}'

            assert x2.shape[1] == self.l2_in_channels, \
                f'layer2 通道數不符: {x2.shape[1]} vs {self.l2_in_channels}'
        

        # 需要哪些 key，以及各自的期望維度
        # 用字典是因為兩個需求方可能要到同一個 key（shared 注入和 gating 都要 motion_L）
        need = {}
        if self.use_gate:
            if self.dual_gate:
                for k in ('motion_upper_L', 'motion_upper_S',
                          'motion_lower_L', 'motion_lower_S'):
                    need[k] = self.motion_dim_part
            else:
                for k in ('motion_L', 'motion_S'):
                    need[k] = self.motion_dim_whole

        if self.motion_inject != 'none':
            if self.inject_scope == 'upper_lower': # 代表 dual_gate 會用到
                for k in ('motion_upper_L', 'motion_upper_S',
                          'motion_lower_L', 'motion_lower_S'):
                    need.setdefault(k, self.motion_dim_part)
            else:
                for k in ('motion_L', 'motion_S'): # 代表 其他都會用到
                    need.setdefault(k, self.motion_dim_whole)

        motion = {}
        if need: # 都不需要就跳過（gating 和注入都不需要的情況)
            assert data_samples is not None, \
                '[Neck] gating / motion_inject 需要 data_samples'
            dev = x3.device if x3 is not None else x2.device
            for k, dim in need.items():
                motion[k] = self._stack_motion(data_samples, k, B, dev, dim)
                    
        # F_L：深層 (layer3)，涵蓋整段 21 幀
        # 針對 H, W 做空間 GAP，再針對 T 做時間 GAP
        F_L = None
        if self.use_lw:
            F_L = self.temporal_pool(self.spatial_pool(x3)).view(B, -1).float()

        F_S = None
        if self.use_sw:
            m, T2 = self.crop_margin, x2.shape[2] # shape: [B, C, T2, H, W]
            assert T2 > 2 * m, f"[Neck] layer2 的 T={T2} 太短"
            x2_small = x2[:, :, m:T2 - m] # [, , 9:12 (21-9), , ]，剛好 15 幀
            if self.debug and not self._printed_once:
                print(f"[Neck DEBUG] 小視窗(幀 {m}~{T2-m-1}): {tuple(x2_small.shape)}")
            F_S = self.small_proj(
                self.temporal_pool(self.spatial_pool(x2_small)).view(B, -1)).float()

        if self.debug and not self._printed_once and F_S is not None and F_L is not None:
            print(f"[Scale] |F_L| {F_L.norm(dim=1).mean().item():.3f}   "
                  f"|F_S| {F_S.norm(dim=1).mean().item():.3f}   "
                  f"ratio {(F_L.norm(dim=1).mean() / F_S.norm(dim=1).mean()).item():.2f}")

        if self.debug and self.training:
            self._step += 1

        def _inject(injector, key_long, key_short):
            """把 F_L 配 motion_L、F_S 配 motion_S 各自注入，不跨視窗混合"""
            feats, motions, slots = [], [], []
            # 收集當前模式下實際存在的視覺特徵張量（F_L 或 F_S）
            # 收集與 feats 嚴格成對、一一對應的運動特徵
            # 屬於長視窗（'L'）還是短視窗（'S'）
            if F_L is not None: 
                feats.append(F_L)
                motions.append(motion[key_long])
                slots.append('L')
            if F_S is not None:
                feats.append(F_S)
                motions.append(motion[key_short])
                slots.append('S')
            done = dict(zip(slots, injector(feats, motions)))
            return done.get('L'), done.get('S')

        F_up = F_low = None
        if self.motion_inject != 'none': # 長短自己的運動特徵參數，不是原本的控制比例用的
            # inject 裡面自己就會把前面的 F_L、F_S 依照長短視窗各自注入
            if self.inject_scope == 'whole_body':
                F_L, F_S = _inject(self.inject_whole, 'motion_L', 'motion_S') 

            else: # 走到這裡代表選了「上下半身各用自己的運動特徵注入」(mode = dual_gate)
                F_L_up, F_S_up = _inject(
                    self.inject_upper, 'motion_upper_L', 'motion_upper_S')
                F_L_low, F_S_low = _inject(
                    self.inject_lower, 'motion_lower_L', 'motion_lower_S')

                # F_L_up, F_S_up 是上半身的長短視窗特徵
                # F_L_low, F_S_low 是下半身的長短視窗特徵
                if self.mode != 'dual_gate':
                    raise ValueError(
                        f"[Neck] inject_scope='upper_lower' 只支援 dual_gate，"
                        f"收到 {self.mode}。其餘 mode 請用 'whole_body'")

                F_up = self._gate_fuse(
                    self.motion_scorer_upper,
                    motion['motion_upper_L'], motion['motion_upper_S'],
                    F_L_up, F_S_up, tag=' upper')
                F_low = self._gate_fuse(
                    self.motion_scorer_lower,
                    motion['motion_lower_L'], motion['motion_lower_S'],
                    F_L_low, F_S_low, tag=' lower')


        if F_up is not None:
            pass  # (mode = dual_gate 且 motion_inject != 'none')，已經在上面做完 gating

        elif self.mode == 'dual_gate':
            F_up = self._gate_fuse(
                self.motion_scorer_upper,
                motion['motion_upper_L'], motion['motion_upper_S'],
                F_L, F_S, tag=' upper')
            F_low = self._gate_fuse(
                self.motion_scorer_lower,
                motion['motion_lower_L'], motion['motion_lower_S'],
                F_L, F_S, tag=' lower')

        elif self.mode == 'per_gate':
            F_up = F_low = self._gate_fuse(
                self.motion_scorer,
                motion['motion_L'], motion['motion_S'], F_L, F_S)

        elif self.mode == 'dual_window':
            F_up = F_low = 0.5 * F_L + 0.5 * F_S

        elif self.mode == 'learnable':
            alpha = torch.sigmoid(self.alpha_logit).view(1, -1)
            F_up = F_low = alpha * F_L + (1.0 - alpha) * F_S

        elif self.mode == 'long_only':
            F_up = F_low = F_L

        elif self.mode == 'short_only':
            F_up = F_low = F_S

        F_up = F_up.view(B, self.out_channels, 1, 1, 1)
        F_low = F_low.view(B, self.out_channels, 1, 1, 1)

        if self.debug and not self._printed_once:
            print(f"[DualWindowGatingNeck DEBUG] F_up {tuple(F_up.shape)}  "
                  f"F_low {tuple(F_low.shape)}\n")

        self._printed_once = True

        # 一律回傳 (上半身特徵, 下半身特徵)；非 dual 模式兩者是同一個張量
        return (F_up, F_low), dict()

    def _gate_fuse(self, scorer, motion_L, motion_S, F_L, F_S, tag=''):
        """長短視窗的 motion 經 scorer → 跨視窗 softmax → 逐通道融合"""
        # 沿 batch 維 concat，讓長短視窗共用同一份 BN 統計
        motion_pair = torch.cat([motion_L, motion_S], dim=0)
        score_pair = scorer(motion_pair)
        score_L, score_S = score_pair.chunk(2, dim=0)

        weights = torch.softmax(
            torch.stack([score_L, score_S], dim=1), dim=1)
        alpha, beta = weights[:, 0], weights[:, 1]   # 各為 [B, C]

        if self.debug and (not self._printed_once or
                           (self.training and self._step % 200 == 0)):
            a = alpha.detach().float()
            print(f"[Gating{tag}] step {self._step}  "
                  f"alpha mean {a.mean().item():.4f}  "
                  f"跨通道 std {a.std(dim=1).mean().item():.4f}  "
                  f"跨樣本 std {a.mean(dim=1).std().item():.4f}")

        return alpha * F_L + beta * F_S
    
    def _stack_motion(self, data_samples, key, batch_size, device, expect_dim):
        motion_list = [
            torch.as_tensor(
                getattr(sample, key),
                dtype=torch.float32,
                device=device
            ).flatten()
            for sample in data_samples
        ]

        motion = torch.stack(motion_list, dim=0)

        assert motion.shape[1] == expect_dim, \
            f"{key} 維度錯誤：{motion.shape[1]} vs {expect_dim}"

        assert batch_size % motion.shape[0] == 0, \
            f"batch size {batch_size} 無法對應 {motion.shape[0]} 筆 motion"

        if motion.shape[0] != batch_size:
            repeat = batch_size // motion.shape[0]
            motion = motion.repeat_interleave(repeat, dim=0)

        return motion