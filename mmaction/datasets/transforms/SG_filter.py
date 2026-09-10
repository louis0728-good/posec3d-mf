import numpy as np
from scipy.signal import savgol_filter
from mmaction.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class SG(BaseTransform):
    def __init__(self, crop_margin=3, window_length=5, polyorder=2, num_person=1, upper_kp=None, lower_kp=None):
        """
        crop_margin: 小視窗要從大視窗前後各裁掉幾幀 (依據圖，21-15=6，單邊裁 3 幀)
        """
        self.crop_margin = crop_margin
        self.window_length = window_length
        self.polyorder = polyorder
        self.num_person = num_person
        self.upper_kp = list(self.UPPER_KP if upper_kp is None else upper_kp)
        self.lower_kp = list(self.LOWER_KP if lower_kp is None else lower_kp)
        assert not (set(self.upper_kp) & set(self.lower_kp)), \
            '[SG] upper_kp 與 lower_kp 不可重疊'

        # 給 config 對照用，避免 motion_feat_dim 手動填錯
        self.dim_upper = len(self.upper_kp) * 6 * num_person   # 預設 36
        self.dim_lower = len(self.lower_kp) * 6 * num_person   # 預設 36

    SHOULDER = [5, 6]      # COCO-17 左右肩
    HIP      = [11, 12]    # COCO-17 左右髖
    # 上半身：肩、肘、腕（排除鼻/眼/耳 0~4）
    UPPER_KP = [5, 6, 7, 8, 9, 10]
    # 下半身：髖、膝、踝
    LOWER_KP = [11, 12, 13, 14, 15, 16]

    def _body_scale(self, kps):
        """kps: [M, T, V, C] → 回傳 [M, 1, 1, 1] 的軀幹長度"""
        sh = kps[:, :, self.SHOULDER, :].mean(axis=2)    # [M, T, C] 肩中點
        hp = kps[:, :, self.HIP, :].mean(axis=2)         # [M, T, C] 髖中點
        d  = np.linalg.norm(sh - hp, axis=-1)            # [M, T]
        s  = np.median(d, axis=1)                        # [M] 用中位數抗離群
        return np.maximum(s, 1e-3).reshape(-1, 1, 1, 1)
    
    def _fix_num_person(self, feat):
        """固定人數維度，不夠補零，多的裁掉"""
        # feat shape: [M, V] = [1, 6]
        M = feat.shape[0]
        if M >= self.num_person: # 若實際 M ≥ num_person：只取 1 個人
            return feat[:self.num_person] 
        else:
            pad = np.zeros((self.num_person - M,) + feat.shape[1:], dtype=feat.dtype) # 決定新陣列的 shape
            # pad = (0, 17, 2) = 無效陣列，不會抱錯但就是沒有更動
            return np.concatenate([feat, pad], axis=0) # 若實際 M < num_person：補零到 num_person
    
    def _apply_sg(self, kps): # kps 預期 shape [M, T, V, C]
        # m = 人 t = 時間 v = 關節 c = 座標
        """共用的 S-G 計算邏輯，保留每個選手的個別特徵"""
        T = kps.shape[1] # shape: [M, T, V, C]
        wl = min(self.window_length, T) # 理論上應該會是 5，T 應該會大於 5
        if wl % 2 == 0: wl -= 1 # 理論上不會觸發
        # window_length: 在時間序列上，看「附近幾幀」的資料，做平滑，減少抖動。
        M, _, V, C = kps.shape
        if wl <= self.polyorder + 1:                                    # ← 補這兩行
            return np.zeros((self.num_person * V * 6,),dtype=np.float32) # 若視窗太小，無法做 S-G，直接回傳零向量
        
        # crop_margin: 從長視窗的前後各裁掉 3 幀，得到短視窗。
        # window_length: S-G 濾波器的滑動視窗大小(一次看多少來濾)。數字越大，濾波越平滑但細節越少。必須是奇數。
        # polyorder: 2 表示用二次多項式(除了速度還有加速度)
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
        sp = np.linalg.norm(vel, axis=-1)      # [M, T, V] 每幀每關節的速度大小
        am = np.linalg.norm(acc, axis=-1)      # [M, T, V] 加速度大小

        mean_velocity = vel.mean(axis=1)
        mean_speed = sp.mean(axis=1)
        net_speed = np.linalg.norm(mean_velocity, axis=-1)
        straightness = net_speed / (mean_speed + 1e-6)

        feats = (
            mean_speed, # [M, V] 每個關節的平均速度
            sp.std(axis=1), # [M, V] 每個關節的速度標準差
            sp.max(axis=1),   # [M, V] 速度峰值（爆發）      ← 加回
            am.mean(axis=1), # [M, V] 每個關節的平均加速度
            am.max(axis=1),   # [M, V] 加速度峰值
            straightness, # [M, V] 每個關節的平均直線度 (0~1，越接近 1 越直)
        )

        feats = [self._fix_num_person(f) for f in feats]
        return np.concatenate(
            [f.flatten() for f in feats]
        ).astype(np.float32)

    def transform(self, results: dict) -> dict:
        """ result:
            'frame_dir': 'test001_p1',
            'label: [label_upper, label_lower]
            'img_shape': (1080, 1920),
            'original_shape': (1080, 1920),
            'total_frames': 150,
            'keypoint': np.array(...),       # (1, T, 17, 2)
            'keypoint_score': np.array(...)  # (1, T, 17)
        """
        kps = results['keypoint'].astype(np.float32) # shape: [M, T, V, C]
        assert kps.shape[2] == 17, \
            f'[SG] 預期 COCO-17，收到 V={kps.shape[2]}'

        # 尺度正規化一定要在「全 17 點」上做，
        # 因為 _body_scale 用的是肩(5,6) / 髖(11,12)，切完就找不到了
        kps = kps / self._body_scale(kps) 
        T = kps.shape[1]

        # 短視窗：前後各裁掉 crop_margin 幀（21 → 15）
        s = self.crop_margin
        e = max(T - self.crop_margin, s + 1)
        kps_small = kps[:, s:e, :, :]


       # 全身版本：保留舊 key，讓「單一 gate」那組消融不用重跑 pipeline 就能比
        body_kp = self.upper_kp + self.lower_kp
        results['motion_L'] = self._apply_sg(kps[:, :, body_kp, :])
        results['motion_S'] = self._apply_sg(kps_small[:, :, body_kp, :])

        # 上/下半身分開算
        results['motion_upper_L'] = self._apply_sg(kps[:, :, self.upper_kp, :]) # [M, T, 6, C]
        results['motion_upper_S'] = self._apply_sg(kps_small[:, :, self.upper_kp, :])
        results['motion_lower_L'] = self._apply_sg(kps[:, :, self.lower_kp, :])
        results['motion_lower_S'] = self._apply_sg(kps_small[:, :, self.lower_kp, :])

        return results