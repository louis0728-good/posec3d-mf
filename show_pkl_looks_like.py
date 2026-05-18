import mmengine
import numpy as np
import random

pkl_path = r'pkl/ntu60_2d.pkl'
data = mmengine.load(pkl_path)

print(' keys: ', list(data.keys()))
print('\nsplit: ', data['split'])
anno = random.choice(data['annotations'])
print('\nannotation keys: ', list(anno.keys()))

print(f'\nframe_dir:      {anno["frame_dir"]}')
print(f'label:          {anno["label"]}')
print(f'img_shape:      {anno["img_shape"]}')
print(f'total_frames:   {anno["total_frames"]}')
print(f'keypoint shape: {anno["keypoint"].shape}')         # 預期 (2, T, 17, 2)
print(f'kp_score shape: {anno["keypoint_score"].shape}')   # 預期 (2, T, 17)

# 看前 3 幀的數據
for pid in range(anno['keypoint'].shape[0]):
    print(f'\nPerson {pid+1} 前 10 幀 ')
    for f in range(min(10, anno['total_frames'])):
        kp = anno['keypoint'][pid, f]       # (17, 2)
        sc = anno['keypoint_score'][pid, f]  # (17,)
        is_empty = '(空)' if np.all(kp == 0) else ''
        print(f'  幀 {f}: nose=({kp[0][0]:.0f}, {kp[0][1]:.0f})  '
              f'score={sc[0]:.2f}  {is_empty}')