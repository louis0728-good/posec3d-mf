import numpy as np
from mmaction.datasets.transforms.SG_filter import SG

kps = np.random.randn(1, 21, 17, 2).astype(np.float32) * 50 + 500
r = SG(crop_margin=3, window_length=5, polyorder=2).transform(
    {'keypoint': kps})

for k in ('motion_L', 'motion_S',
          'motion_upper_L', 'motion_upper_S',
          'motion_lower_L', 'motion_lower_S'):
    print(f'{k:16s} {r[k].shape}  {r[k].dtype}  '
          f'finite={np.isfinite(r[k]).all()}')