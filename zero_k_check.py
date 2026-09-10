import pickle, numpy as np
d = pickle.load(open('pkl/train_val.pkl','rb'))
z = t = 0
for a in d['annotations']:
    kp = a['keypoint']                       # (M,T,V,2)
    z += int((np.abs(kp).sum(-1) == 0).sum())
    t += kp.shape[0] * kp.shape[1] * kp.shape[2]
print(f'零座標關節: {z}/{t} = {z/t:.2%}')