import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class DualWindowGatingNeck(BaseModule):
    def __init__(self, in_channels, sg_feat_dim, out_channels=512, 
                 crop_ratio=0.7, debug=False):
        super().__init__()
        self.in_channels = in_channels
        self.sg_feat_dim = sg_feat_dim
        self.out_channels = out_channels
        self.debug = debug
        self._printed_once = False

        # 決定小視窗要從 Feature Map 的 T 維度哪裡切到哪裡
        # 假設大視窗是 21 幀 (t-10 ~ t+10)，小視窗是 15 幀 (t-7 ~ t+7)
        self.crop_ratio = crop_ratio # crop_ratio=0.7 這樣不知道有沒有降採樣 (有)

        # 空間池化 (GAP)
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # T 維保持一樣，後面的 h, w 變成 1
        # [B, C, T, H, W] -> [B, C, T, 1, 1]

        # 時間池化 (Temporal Pooling)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) # 跟上面差不多
        """
            data_sample = gt_label、sg_features ...
            gt_label = (results['label'])
        """
        # Gating Machine (將 F_L, F_S, sg_features 融合)
        # 總輸入維度 = Backbone通道(F_L) + Backbone通道(F_S) + S-G向量維度
        # 大小視窗都 512 維的情況?
        fused_dim = self.in_channels * 2 + self.sg_feat_dim  # 這裡應該就是 gin 的總維度了
        # 所以理想狀況應該會是 512 * 2 + 136 =1160
        """ 要去 sg 做確認，dim 維度 """
        """
        fused_dim 這行在算輸入維度：
                F_L：一個 C 維向量
                F_S：另一個 C 維向量
                上面的都是剛剛 BACKBONE 的輸出

                sg_features：一個 sg_feat_dim 維向量(S-G 輸出的骨架的速度加速度)
        """
        """
            data_sample = gt_label、sg_features ...
            gt_label = (results['label'])
        """
        # 算 alpha beta 權重
        hidden_dim = out_channels // 2 # 降維
        self.gating_machine = nn.Sequential( # MLP
            nn.Linear(fused_dim, hidden_dim), # 第一層全連接層 (input, output)
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )
        self.softmax = nn.Softmax(dim=1) # 確保兩權重相加為一

    def forward(self, x, data_samples=None, **kwargs): # 接收參數改為 data_samples
        # x shape: [B, C, T, H, W]
        #          [B, 512, 11, H, W]
        B, C, T, H, W = x.shape
        if self.debug and not self._printed_once:
            print(f"\n[DualWindowGatingNeck DEBUG] 輸入 Feature Map shape: {x.shape}")

        assert C == self.in_channels, f"通道數不符: {C} vs {self.in_channels}"
        assert data_samples is not None, "[Neck] 錯誤：data_samples 為 None，去確認 Recognizer(應該是 custom_recognizer.py) 有傳遞過來！"
        # 呼應前面的 loss, predict

        sg_feats_list = []
        # 原始 batch 裡有幾個樣本，就有幾個 data_sample
        # 但 feature 的 B 不一定等於原始樣本數，因為 [N, num_clips, C, T, H, W] -> [N * num_clips, C, T, H, W]
        for sample in data_samples:
            assert hasattr(sample, 'sg_features'), \
                "[Neck] 錯誤：找不到 'sg_features'！檢查 PackActionInputs 的 algorithm_keys=('sg_features',)"
        
            # 把 numpy 轉成 tensor，並確保跟 x 在同一張顯卡上
            feat_tensor = torch.as_tensor(sample.sg_features, dtype=torch.float32, device=x.device) # 不知道會不會有問題，不行就改回 tensor()
            # data_sample.sg_features 是在 formatting 存上的
            # sg_features 應該會是 136 維
            sg_feats_list.append(feat_tensor)
            
        sg_features = torch.stack(sg_feats_list) # shape: [B, sg_feat_dim] 
        # 假設 N*num_clips = 16 那這裡的batch 應該就會是 16(clips=1 我設定的)。
        # 那 sg_features 應該會長 [16, 136]

        # 測試時 num_clips > 1，B = N * num_clips，需要 repeat
        if sg_features.shape[0] != B: # 通常不會進入 因為我的 num_clips 設定為 1 
            # 跟那個 CLIP/CROP 有關
            num_crops = B // sg_features.shape[0]
            # 這個 我先不加，因為不太會觸發 if not self._printed_once:
            print(
                f"[dual_window_gating][DEBUG] 偵測到 batch 不一致，準備展開 sg_features："
                f"B={B}, sg_batch={sg_features.shape[0]}, num_crops={num_crops}。"
                f"這通常發生在 num_clips>1，因為同一筆 data_sample 會對應多個 views。"
                f"順帶一提， sg_feature shape 大概會長 : shape: [{B}, sg_feat]。"
            )
            sg_features = sg_features.unsqueeze(1).expand(
                -1, num_crops, -1).reshape(B, -1)
            if self.debug and not self._printed_once:
                print(f"[DualWindowGatingNeck DEBUG] sg_features repeat {num_crops}x → {sg_features.shape}")
        elif not self._printed_once: 
            print(
                f"[dual_window_gating][DEBUG] B == sg_features.shape[0]，跳過展開："
                f"B={B}, sg_batch={sg_features.shape[0]}。"
                f"通常表示 num_clips=1，每個 data_sample 只對應一筆 feature。"
            )
            
        # --- 產生 F_L (大視窗特徵) ---
        # 針對 H, W 做空間 GAP，再針對 T 做時間 GAP
        F_L = self.temporal_pool(self.spatial_pool(x)).view(B, -1) # shape: [B, C]
        # 這是用整個 feature map 的時間長度 T 做平均，得到整個大視窗的全局向量。

        # --- 產生 F_S (小視窗特徵) ---
        crop_len = max(1, int(T * self.crop_ratio)) # backbone 降採樣後應該剩下 21/2 = 11, 假設 11* 0.7 = ７
        start_idx = (T - crop_len) // 2 # 11-7=4, 4//2==2
        end_idx = start_idx + crop_len # = 2+7==9

        x_small = x[:, :, start_idx:end_idx, :, :] #　包含頭，不包含尾 [2,3,4,5,6,7,8]
        if self.debug and not self._printed_once:
            print(f"[DualWindowGatingNeck DEBUG] 裁切後的小視窗 shape(注意這裡已經是被降採樣過了): {x_small.shape}")

        F_S = self.temporal_pool(self.spatial_pool(x_small)).view(B, -1) # shape: [B, C]

        # --- Gating Fusion ---
        g_in = torch.cat([F_L, F_S, sg_features], dim=1) 

        # MLP 產生 2 個 Logits
        abc = self.gating_machine(g_in)

        # Softmax 產生 alpha, beta (相加為 1)
        gating_weights = self.softmax(abc)
        alpha = gating_weights[:, 0:1] # shape: [B, 1]
        beta = gating_weights[:, 1:2]  # shape: [B, 1]
        
        if self.debug and not self._printed_once:
            print(f"[Gating] Alpha (大視窗權重): {alpha[0].item():.4f}, Beta (小視窗權重): {beta[0].item():.4f}")

        F_fused = alpha * F_L + beta * F_S

        # 為了能順利送進 Head
        F_fused = F_fused.view(B, self.out_channels, 1, 1, 1)

        if self.debug and not self._printed_once:
            print(f"[DualWindowGatingNeck DEBUG] 輸出的 F_fused shape: {F_fused.shape}\n")

        # 跑完第一次 forward 後，永遠關閉列印開關
        self._printed_once = True
    
        # 回傳 Tuple，第二個位置留給空的 aux_loss 字典 (MMAction2 標準格式)
        return F_fused, dict() # 只要最後 head 有算 loss，整條鏈上所有可訓練參數都會被 autograd 照顧到。不用特地傳 alpha, beta 的 loss(應該)