import pickle, numpy as np
d = pickle.load(open('pkl/train_val.pkl','rb'))
tf = np.array([a['total_frames'] for a in d['annotations']])
print('n      =', len(tf))
print('min    =', tf.min(), ' max =', tf.max())
print('分位數 =', np.percentile(tf, [5,25,50,75,90,95,99]).round(1))
print('Δt(=T/21) 分位數 =', (np.percentile(tf,[5,50,95])/21).round(2))
for w in (21, 31, 41, 51, 61):
    print(f'clip_len={w:3d} → 會 wrap-around 的樣本數 = {(tf < w).sum():5d}')