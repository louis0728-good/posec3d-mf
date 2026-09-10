from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmaction.registry import DATASETS
import collections
import mmengine

init_default_scope('mmaction')
cfg = Config.fromfile('configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py')

# =========================================================
# 1. 統計「視窗」數量（模型實際看到的）
# =========================================================
ds = DATASETS.build(cfg.train_dataloader.dataset.dataset)  # 剝掉 RepeatDataset
cu, cl = collections.Counter(), collections.Counter()
for i in range(len(ds)):
    lab = ds.get_data_info(i)['label']
    cu[lab[0]] += 1
    cl[lab[1]] += 1

# =========================================================
# 2. 統計「原始 clip」數量（切窗前）
# =========================================================
pkl = mmengine.load(cfg.train_dataloader.dataset.dataset.ann_file)
ann = {a['frame_dir']: a for a in pkl['annotations']}

cu_clip, cl_clip = collections.Counter(), collections.Counter()
for name in pkl['split']['xsub_train']:
    a = ann[name]
    cu_clip[int(a['label_upper'])] += 1
    cl_clip[int(a['label_lower'])] += 1

# =========================================================
# 3. 印出結果
# =========================================================
def show(title, c):
    n, mx = sum(c.values()), max(c.values())
    print(f'\n=== {title}（共 {n}）===')
    for i in sorted(c):
        print(f'  class {i}: {c[i]:6d}  {c[i]/n:6.2%}  →  權重 {mx/c[i]:.2f}')
    print('class_weight =', [round(mx / c[i], 2) for i in sorted(c)])

show('upper 視窗', cu)
show('lower 視窗', cl)
show('upper 原始 clip', cu_clip)
show('lower 原始 clip', cl_clip)