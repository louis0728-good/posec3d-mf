import numpy as np
from scipy.signal import savgol_filter
from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

"""
從 results['keypoint']（骨架座標）計算「速度/加速度」摘要特徵，
最後塞進 results['sg_features']，讓後面的 PackActionInputs(algorithm_keys=('sg_features',)) 
# PackActionInputs 在 formatting.py
打包進 data_samples
"""
@TRANSFORMS.register_module()
class SG(BaseTransform):
    def __init__(self, crop_margin=3, window_length=3, polyorder=2, num_person=1):
        """
        crop_margin: 小視窗要從大視窗前後各裁掉幾幀 (依據圖，21-15=6，單邊裁 3 幀)
        """
        self.crop_margin = crop_margin
        self.window_length = window_length
        self.polyorder = polyorder
        self.num_person = num_person

    def _fix_num_person(self, feat):
        """固定人數維度，不夠補零，多的裁掉"""
        # feat shape: [M, V, C] = [1, 12, 2]
        M = feat.shape[0]
        if M >= self.num_person: # 若實際 M ≥ num_person：只取 1 個人
            return feat[:self.num_person] 
        else: # 我會觸發這個
            pad = np.zeros((self.num_person - M,) + feat.shape[1:], dtype=feat.dtype) # 決定新陣列的 shape
            # pad = (0, 12, 2) = 無效陣列，不會抱錯但就是沒有更動
            return np.concatenate([feat, pad], axis=0) # 若實際 M < num_person：補零到 num_person
    
    def _apply_sg(self, kps): # kps 預期 shape [M, T, V, C]
        """共用的 S-G 計算邏輯，保留每個選手的個別特徵"""
        T = kps.shape[1] # shape: [M, T, V, C]
        wl = min(self.window_length, T) # 理論上應該會是 3，T 應該會大於 3
        if wl % 2 == 0: wl -= 1 # 理論上不會觸發
        # window_length: 在時間序列上，看「附近幾幀」的資料，做平滑，減少抖動。
        M, _, V, C = kps.shape

        # crop_margin: 從長視窗的前後各裁掉 3 幀，得到短視窗。
        # window_length: S-G 濾波器的滑動視窗大小(一次看多少來濾)。數字越大，濾波越平滑但細節越少。必須是奇數。
        # polyorder: 2 表示用二次多項式(除了速度還有加速度)
        if wl <= self.polyorder:
            return np.zeros((self.num_person * V * C * 2,), dtype=np.float32)
        # SG filter 的條件是 window_length > polyorder，否則不能做。
        """
        如果多項式階數是 polyorder = p： y = a0 + a1 x + a2 x² + ... + ap x^p
        需要求的參數數量是： p + 1 個，簡單講就是視窗大小至少要 > P
        """
        # deriv=1: 取 一階導數，以此類推。
        vel = savgol_filter(kps, window_length=wl, polyorder=self.polyorder, deriv=1, axis=1)
        acc = savgol_filter(kps, window_length=wl, polyorder=self.polyorder, deriv=2, axis=1) 
        # axis = 1 代表沿著時間維度做濾波，[N, T, V, C]

        # 只對時間軸取平均，保留 M 維度 → [M, V, C]
        vel_feat = vel.mean(axis=1)  # [M, V, C] [1, 12, 2]
        acc_feat = acc.mean(axis=1)  # [M, V, C] [1, 12, 2]

        # 固定 M 維度
        vel_feat = self._fix_num_person(vel_feat)  # [num_person, V, C]
        acc_feat = self._fix_num_person(acc_feat)  # [num_person, V, C]

        return np.concatenate([vel_feat.flatten(), acc_feat.flatten()])

    def transform(self, results: dict) -> dict:
        """ result:
            'frame_dir': 'test001_p1',
            'label: [label_upper, label_lower]
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 150,
            'keypoint': np.array(...),       # (1, T, 17, 2)
            'keypoint_score': np.array(...)  # (1, T, 17)
            'sg_features' = concat(feat_l, feat_s) # 經由sg_filter' 新增 sg_feature
        """
        kps = results['keypoint'] # shape: [M, T, V, C]
        T = kps.shape[1]

        # 大視窗 S-G 提取 (全部幀)
        feat_L = self._apply_sg(kps)

        # 小視窗 S-G 提取 (中間段)
        start_idx = self.crop_margin # 3
        end_idx = max(T - self.crop_margin, start_idx + 1) # 防呆，避免切到變空陣列 , 21 - 3 = 18, 3+1。還是 18 大
        kps_small = kps[:, start_idx:end_idx, :, :] # kps[:, 3:18, :, :] 剛好 15
        feat_S = self._apply_sg(kps_small)

        # 3. 將大小視窗特徵 Concat 在一起
        sg_features = np.concatenate([feat_L, feat_S])
        
        # 存入 results 字典
        results['sg_features'] = sg_features.astype(np.float32)
        return results