# -*- coding: utf-8 -*-
"""
make_report.py — 把 work_dirs 的訓練 log 與 VAL 的診斷結果整理成消融報告與趨勢圖

只讀檔，不載入模型，也不改任何訓練或模型程式碼。在 E:\\rcnn\\mmaction2 底下執行：
    python VAL/make_report.py

資料來源
  work_dirs/**/vis_data/scalars.json   每個 run 的訓練 loss（每 20 iter 一筆）與驗證準確率（每 epoch 一筆）
  VAL/diag_<tag>_seed<n>.json + .npz   tt2.py 對 best checkpoint 的逐類別結果與逐視窗預測
  VAL/abl_<tag>_seed<n>.json           tt2.py 的反事實消融（沒加 --no-ablation 才會有）

輸出（VAL/report/）
  fig1_overview.png         全部設定 × 每個 seed 的準確率一覽（兩種選 epoch 方式）
  fig2_forest.png           逐題回答「這個設定有沒有用」：配對差異與 95% 信賴區間
  fig3a~3d_*.png            訓練趨勢比較，每張只放 2～3 條線（3 seed 平均 + 範圍陰影）
  fig4_counterfactual.png   反事實消融：推論時關掉某個成分會掉多少
  fig5_reliance.png         「模型有在用它」與「模型需要它」的對照
  all_settings_acc.png / all_settings_loss.png / per_setting/*.png
  runs.csv / settings.csv / comparisons.csv / ablations.csv

兩種證據的差別（報告的核心）
  重新訓練比較：A 設定和 B 設定各自從頭訓練、同 seed 相減。這才回答「加了有沒有比較好」。
  反事實消融  ：同一個訓練好的模型，推論時把某成分關掉。這只回答「模型有沒有在用它」，
                關掉會掉分不代表加了它比較好——模型沒有它時會學別的路徑補回來。
"""

import argparse
import collections
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ══════════════════════════════════════════════════════════════════
#  設定清單
# ══════════════════════════════════════════════════════════════════
# (key, work_dirs 資料夾名（去掉日期與 seed）, (neck mode, motion_inject), 圖上名稱)
SETTINGS = [
    ('long',    '純長視窗',                 ('long_only', 'none'),    '純長視窗'),
    ('short',   '純短視窗',                 ('short_only', 'none'),   '純短視窗'),
    ('dw',      '無門控_雙視窗',             ('dual_window', 'none'),  '無門控雙視窗'),
    ('learn',   '隨機學習',                 ('learnable', 'none'),    '隨機學習'),
    ('pg',      '單門控',                   ('per_gate', 'none'),     '單門控'),
    ('dg',      '雙門控',                   ('dual_gate', 'none'),    '雙門控'),
    ('long_b',  '純長視窗_運動參數beta',      ('long_only', 'beta'),    '純長視窗+β'),
    ('short_b', '純短視窗_運動參數beta',      ('short_only', 'beta'),   '純短視窗+β'),
    ('dw_b',    '無門控_雙視窗_運動參數beta',  ('dual_window', 'beta'),  '無門控雙視窗+β'),
    ('pg_b',    '單門控_運動參數beta',        ('per_gate', 'beta'),     '單門控+β'),
    ('dg_b',    '雙門控_運動參數beta',        ('dual_gate', 'beta'),    '雙門控+β'),
    ('dg_g',    '雙門控_運動參數gamma',       ('dual_gate', 'gamma'),   '雙門控+γ'),
]
ORDER = [s[0] for s in SETTINGS]
NAME2KEY = {s[1]: s[0] for s in SETTINGS}
MODE2KEY = {s[2]: s[0] for s in SETTINGS}
LABEL = {s[0]: s[3] for s in SETTINGS}
FOLDER = {s[0]: s[1] for s in SETTINGS}

# 每一題：(問題, [(實驗組, 對照組), ...], 算式)。多組配對時合併成一個平均差異
QUESTIONS = [
    ('只用短視窗可以嗎？',        [('short', 'long')], '純短視窗 − 純長視窗'),
    ('加上短視窗有幫助嗎？',      [('dw', 'long')],    '無門控雙視窗 − 純長視窗'),
    ('融合權重用學的有幫助嗎？',  [('learn', 'dw')],   '隨機學習 − 無門控雙視窗'),
    ('用運動特徵做門控有幫助嗎？', [('pg', 'dw')],     '單門控 − 無門控雙視窗'),
    ('上下半身分開門控有幫助嗎？', [('dg', 'pg')],     '雙門控 − 單門控'),
    ('注入運動參數 β 有幫助嗎？',
     [('long_b', 'long'), ('dw_b', 'dw'), ('pg_b', 'pg'), ('dg_b', 'dg')],
     '+β − 無注入（4 種架構合併）'),
    ('注入運動參數 γ 有幫助嗎？',  [('dg_g', 'dg')],   '雙門控+γ − 雙門控'),
    ('最完整的設定贏過基準嗎？',  [('dg_g', 'long')],  '雙門控+γ − 純長視窗'),
]
# 只放表格、不上森林圖的補充比較
EXTRA = [
    ('單門控 vs 隨機學習（逐樣本自適應 vs 靜態權重）', [('pg', 'learn')], '單門控 − 隨機學習'),
    ('單門控 vs 基準', [('pg', 'long')], '單門控 − 純長視窗'),
    ('雙門控 vs 基準', [('dg', 'long')], '雙門控 − 純長視窗'),
    ('β：純長視窗', [('long_b', 'long')], '純長視窗+β − 純長視窗'),
    ('β：無門控雙視窗', [('dw_b', 'dw')], '無門控雙視窗+β − 無門控雙視窗'),
    ('β：單門控', [('pg_b', 'pg')], '單門控+β − 單門控'),
    ('β：雙門控', [('dg_b', 'dg')], '雙門控+β − 雙門控'),
    ('β：純短視窗（只有 seed 44）', [('short_b', 'short')], '純短視窗+β − 純短視窗'),
    ('γ vs β（雙門控）', [('dg_g', 'dg_b')], '雙門控+γ − 雙門控+β'),
]

ABL_NAMES = collections.OrderedDict([
    ('no_FS',           '拿掉短視窗特徵 F_S'),
    ('no_inject',       '關掉運動參數注入'),
    ('inject_L_off',    '只關長視窗的注入'),
    ('inject_S_off',    '只關短視窗的注入'),
    ('no_motion',       '運動特徵全換成平均值'),
    ('motion_L_off',    '只換掉長視窗運動特徵'),
    ('motion_S_off',    '只換掉短視窗運動特徵'),
    ('sg_speed_off',    '換掉速度統計'),
    ('sg_acc_off',      '換掉加速度統計'),
    ('sg_straight_off', '換掉直線度'),
    ('gate_half',       '門控固定 0.5／0.5（＝等權）'),
    ('gate_mean',       '門控固定成平均 α'),
])

LOWER_NAMES = ['無', '前進', '後退', '長刺', '飛刺', '前進長刺']

# ══════════════════════════════════════════════════════════════════
#  配色與樣式（dataviz 預設色盤；灰 = 對照組，藍 = 被檢驗的設定，橘 = 第三條線）
# ══════════════════════════════════════════════════════════════════
C_MAIN, C_ALT, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, SURF = '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_REF = MUTED


def setup_style():
    plt.rcParams.update({
        'font.family': ['Microsoft JhengHei', 'Noto Sans CJK TC', 'DejaVu Sans'],
        'font.size': 10,
        'axes.unicode_minus': False,
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'axes.labelcolor': INK2, 'axes.labelsize': 9.5,
        'xtick.color': AXIS, 'ytick.color': AXIS,
        'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.7, 'grid.linestyle': '-',
        'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlecolor': INK,
        'axes.titlelocation': 'left', 'axes.titlepad': 8,
        'legend.frameon': False, 'legend.fontsize': 9.5,
        'lines.linewidth': 2.0, 'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'savefig.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    })


pct = FuncFormatter(lambda v, _: f'{v * 100:.0f}%' if abs(v * 100 - round(v * 100)) < 1e-6
                    else f'{v * 100:.1f}%')


def pct_axis(axis):
    axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    axis.set_major_formatter(pct)
pp = FuncFormatter(lambda v, _: f'{v:+.0f}' if abs(v) > 1e-9 else '0')


# ══════════════════════════════════════════════════════════════════
#  1. 讀訓練 log
# ══════════════════════════════════════════════════════════════════
FOLDER_RE = re.compile(r'^(?:\d{8}_)?(?P<name>.+?)_seed[=_-]?(?P<seed>\d+)$')


def load_logs(work):
    runs = {}
    for sj in sorted(glob.glob(os.path.join(work, '**', 'vis_data', 'scalars.json'),
                            recursive=True)):
        parts = os.path.normpath(sj).split(os.sep)
        m = FOLDER_RE.match(parts[-4])
        if not m:
            print(f'[略過] 資料夾名稱不符 <設定>_seed=<n>：{sj}')
            continue
        key = NAME2KEY.get(m.group('name'))
        if key is None:
            print(f'[略過] 不在 SETTINGS 清單內的設定「{m.group("name")}」：{sj}')
            continue
        seed = int(m.group('seed'))

        tr, va = collections.defaultdict(list), {}
        with open(sj, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                if 'loss' in d:
                    tr[d['epoch']].append(d)
                elif 'dual_acc/mean1' in d:
                    va[d['step']] = d
        eps = sorted(va)
        if not eps:
            print(f'[略過] 沒有驗證紀錄：{sj}')
            continue
        run_dir = os.path.dirname(os.path.dirname(os.path.dirname(sj)))
        if os.path.normpath(os.path.dirname(run_dir)) != os.path.normpath(work):
            print(f'[注意] {parts[-4]} 被放在另一個 run 的資料夾裡：'
                  f'{os.path.relpath(run_dir, work)}（照樣讀取；run_all.py 不會掃到它）')
        run = dict(
            key=key, seed=seed, path=sj, folder=parts[-4],
            epoch=np.array(eps),
            mean1=np.array([va[e]['dual_acc/mean1'] for e in eps]),
            lower=np.array([va[e]['dual_acc/lower_mean1'] for e in eps]),
            upper=np.array([va[e]['dual_acc/upper_mean1'] for e in eps]),
            loss=np.array([np.mean([d['loss'] for d in tr[e]]) for e in eps]),
            loss_lower=np.array([np.mean([d['loss_cls_lower'] for d in tr[e]]) for e in eps]),
            loss_upper=np.array([np.mean([d['loss_cls_upper'] for d in tr[e]]) for e in eps]),
        )
        b = int(np.argmax(run['mean1']))
        run['best_epoch'] = int(run['epoch'][b])
        for m_ in ('mean1', 'lower', 'upper'):
            run['best_' + m_] = float(run[m_][b])
            run['last3_' + m_] = float(run[m_][-3:].mean())
        if (key, seed) in runs and len(runs[(key, seed)]['epoch']) >= len(eps):
            print(f'[注意] {key} seed={seed} 有多份 log，保留 epoch 較多的那份')
            continue
        runs[(key, seed)] = run
    return runs


def seeds_of(runs, key):
    return sorted(s for k, s in runs if k == key)


def stack(runs, key, metric, seeds=None):
    seeds = seeds or seeds_of(runs, key)
    return np.stack([runs[(key, s)][metric] for s in seeds if (key, s) in runs])


# ══════════════════════════════════════════════════════════════════
#  2. 讀 VAL 的 tt2.py 輸出
# ══════════════════════════════════════════════════════════════════
def seed_from(path, d):
    for s in (os.path.basename(path), d.get('tag', ''), d.get('ckpt', '')):
        m = re.search(r'seed[=_-]?(\d+)', str(s))
        if m:
            return int(m.group(1))
    return None


def load_val(val_dir):
    diag, abl = {}, collections.defaultdict(dict)
    files = (sorted(glob.glob(os.path.join(val_dir, 'diag_*.json')))
             + sorted(glob.glob(os.path.join(val_dir, 'abl_*.json'))))
    for f in files:
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
        key = MODE2KEY.get((d.get('mode'), d.get('motion_inject')))
        seed = seed_from(f, d)
        if key is None or seed is None:
            print(f'[略過] 無法對應到設定：{f}')
            continue
        if 'ablation' in d and seed not in abl[key]:
            abl[key][seed] = d
        if (key, seed) not in diag:
            npz = os.path.splitext(f)[0] + '.npz'
            d['_z'] = dict(np.load(npz, allow_pickle=True)) if os.path.exists(npz) else None
            d['_file'] = os.path.basename(f)
            diag[(key, seed)] = d
    return diag, abl


# ══════════════════════════════════════════════════════════════════
#  3. 統計：配對差異 + 兩層 bootstrap（重抽 clip，也重抽 seed）
# ══════════════════════════════════════════════════════════════════
class Boot:
    """以 clip（frame_dir）為單位重抽驗證集。同一個 clip 的多個 window 一起進出，
    所有設定共用同一組重抽，所以相減是「配對」的。"""

    def __init__(self, cid, B, rng):
        uniq, inv = np.unique(np.asarray(cid), return_inverse=True)
        draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
        cnt = np.zeros((B, len(uniq)), np.float32)
        for b in range(B):
            cnt[b] = np.bincount(draws[b], minlength=len(uniq))
        self.W = cnt[:, inv]
        self.cid = np.asarray(cid)
        self.B = B
        self._cache = {}

    def macro(self, tag, P, G, n):
        if tag in self._cache:
            return self._cache[tag]
        G = np.asarray(G).astype(int)
        ok = (np.asarray(P).argmax(1) == G).astype(np.float32)
        onehot = np.eye(n, dtype=np.float32)[G]
        hit = self.W @ (onehot * ok[:, None])
        tot = self.W @ onehot
        with np.errstate(invalid='ignore', divide='ignore'):
            boot = np.nanmean(hit / np.where(tot > 0, tot, np.nan), axis=1)
        point = float(np.mean([ok[G == c].mean() for c in range(n) if (G == c).any()]))
        self._cache[tag] = (point, boot)
        return point, boot


def aligned(za, zb):
    return (len(za['gt_lower']) == len(zb['gt_lower'])
            and bool((za['gt_lower'] == zb['gt_lower']).all())
            and bool((np.asarray(za['cid']) == np.asarray(zb['cid'])).all()))


def compare(pairs, diag, runs, boot, rng):
    units = []
    for t, c in pairs:
        for s in (42, 43, 44, 45, 46):
            dt, dc = diag.get((t, s)), diag.get((c, s))
            if dt is None or dc is None or dt['_z'] is None or dc['_z'] is None:
                continue
            if not aligned(dt['_z'], dc['_z']):
                print(f'[略過] {t}/{c} seed={s} 的驗證集順序不一致')
                continue
            units.append((t, c, s))
    if not units:
        return None
    heads = {}
    for h, n in (('lower', 6), ('upper', 2)):
        pts, bts = [], []
        for t, c, s in units:
            zt, zc = diag[(t, s)]['_z'], diag[(c, s)]['_z']
            pt, bt = boot.macro((t, s, h), zt[f'pred_{h}'], zt[f'gt_{h}'], n)
            pc, bc = boot.macro((c, s, h), zc[f'pred_{h}'], zc[f'gt_{h}'], n)
            pts.append(pt - pc)
            bts.append(bt - bc)
        heads[h] = (np.array(pts), np.array(bts))
    heads['mean1'] = ((heads['lower'][0] + heads['upper'][0]) / 2,
                      (heads['lower'][1] + heads['upper'][1]) / 2)

    U = len(units)
    pick = rng.integers(0, U, size=(boot.B, U))      # 第二層：重抽 seed（配對單位）
    out = dict(units=units)
    for h, (pts, bts) in heads.items():
        dist = np.take_along_axis(bts.T, pick, axis=1).mean(1)
        lo, hi = np.percentile(dist, [2.5, 97.5])
        out[h] = dict(delta=float(pts.mean()), per_unit=pts, lo=float(lo), hi=float(hi),
                      sig=bool(lo > 0 or hi < 0),
                      agree=int(max((pts > 0).sum(), (pts < 0).sum())))
    # 同一題、改用「最後 3 epoch 平均」（不挑 epoch）的差異，從 log 算
    l3 = [runs[(t, s)]['last3_mean1'] - runs[(c, s)]['last3_mean1']
          for t, c, s in units if (t, s) in runs and (c, s) in runs]
    out['last3_mean1'] = dict(delta=float(np.mean(l3)) if l3 else float('nan'),
                              per_unit=np.array(l3))
    return out


def verdict(r, head='lower'):
    x = r[head]
    if not x['sig']:
        return '在雜訊內（無法區分）'
    return '顯著變好' if x['delta'] > 0 else '顯著變差'


# ══════════════════════════════════════════════════════════════════
#  4. 畫圖
# ══════════════════════════════════════════════════════════════════
def band_line(ax, x, Y, color, label, lw=2.0, z=3, ls='-'):
    Y = np.atleast_2d(Y)
    if len(Y) > 1:
        ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.13, lw=0, zorder=z - 1)
    ax.plot(x, Y.mean(0), color=color, lw=lw, ls=ls, label=label, zorder=z)
    return Y.mean(0)


def end_labels(ax, x_end, ends, min_gap_frac=0.07):
    """線尾直接標名稱；會互撞就不標（交給圖例），不做上下硬推。"""
    lo, hi = ax.get_ylim()
    ys = sorted(v for v, _ in ends)
    if len(ys) > 1 and min(np.diff(ys)) < (hi - lo) * min_gap_frac:
        return
    for v, txt in ends:
        ax.annotate(txt, (x_end, v), xytext=(6, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=8.5, color=INK2)
    ax.set_xlim(right=x_end + (x_end - ax.get_xlim()[0]) * 0.02)


def acc_axis(ax, title):
    pct_axis(ax.yaxis)
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.xaxis.set_major_locator(MultipleLocator(2))


def loss_axis(ax, title='訓練 loss（每個 epoch 平均）'):
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.xaxis.set_major_locator(MultipleLocator(2))


def legend_top(fig, handles, labels, y=0.965):
    fig.legend(handles, labels, loc='upper left', ncol=len(handles),
               bbox_to_anchor=(0.01, y), handlelength=1.8, columnspacing=1.6)


def fig_compare(runs, series, title, fname, seeds=None, note=''):
    """一題一張圖：左 = 驗證準確率（mean1），右 = 訓練 loss。series = [(key, color), ...]"""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    handles, labels = [], []
    for key, color in series:
        ss = seeds or seeds_of(runs, key)
        x = runs[(key, ss[0])]['epoch']
        band_line(axes[0], x, stack(runs, key, 'mean1', ss), color, LABEL[key])
        band_line(axes[1], x, stack(runs, key, 'loss', ss), color, LABEL[key])
        handles.append(Line2D([], [], color=color, lw=2.4))
        labels.append(f'{LABEL[key]}（n={len(ss)}）')
    acc_axis(axes[0], '驗證準確率 mean1（上下半身平均類別準確率）')
    loss_axis(axes[1])
    fig.suptitle(title, x=0.01, y=1.05, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    legend_top(fig, handles, labels, y=0.995)
    txt = '實線 = seed 平均；陰影 = 各 seed 的最高～最低'
    fig.text(0.01, -0.04, txt + ('；' + note if note else ''), fontsize=8.5, color=MUTED)
    fig.subplots_adjust(top=0.84, wspace=0.22)
    fig.savefig(fname)
    plt.close(fig)


def fig_motion(runs, fname):
    """運動參數：每個架構一欄，上 = 驗證 mean1，下 = 訓練 loss；灰 = 無注入，藍 = β，橘 = γ"""
    cols = [('long', 'long_b', None), ('dw', 'dw_b', None), ('pg', 'pg_b', None),
            ('dg', 'dg_b', 'dg_g'), ('short', 'short_b', None)]
    fig, axes = plt.subplots(2, len(cols), figsize=(15.5, 6.2), sharey='row')
    for j, (k0, kb, kg) in enumerate(cols):
        ss = sorted(set(seeds_of(runs, k0)) & set(seeds_of(runs, kb)))
        for i, m in enumerate(('mean1', 'loss')):
            ax = axes[i, j]
            x = runs[(k0, ss[0])]['epoch']
            band_line(ax, x, stack(runs, k0, m, ss), C_REF, '無注入')
            band_line(ax, x, stack(runs, kb, m, ss), C_MAIN, '+β')
            if kg:
                band_line(ax, x, stack(runs, kg, m, ss), C_ALT, '+γ')
            ax.xaxis.set_major_locator(MultipleLocator(4))
            if i == 0:
                pct_axis(ax.yaxis)
                n = f'（只有 seed {ss[0]}）' if len(ss) == 1 else f'（n={len(ss)}）'
                ax.set_title(LABEL[k0] + n)
            else:
                ax.set_xlabel('epoch')
        axes[0, 0].set_ylabel('驗證準確率 mean1')
        axes[1, 0].set_ylabel('訓練 loss')
    handles = [Line2D([], [], color=c, lw=2.4) for c in (C_REF, C_MAIN, C_ALT)]
    fig.suptitle('注入運動參數有幫助嗎？（每一欄是一種架構，灰線 = 同架構不注入）',
                 x=0.01, y=1.04, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    legend_top(fig, handles, ['無注入', '+β（F + b）', '+γ（F·(1+g)，只有雙門控）'], y=0.99)
    fig.text(0.01, -0.03, '實線 = seed 平均；陰影 = 各 seed 的最高～最低。'
             '純短視窗+β 只訓練過 seed 44，所以那一欄兩條線都只取 seed 44。',
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(top=0.86, hspace=0.32, wspace=0.08)
    fig.savefig(fname)
    plt.close(fig)


def fig_overview(runs, fname):
    rows = ['long', 'short', 'dw', 'learn', 'pg', 'dg', None,
            'long_b', 'short_b', 'dw_b', 'pg_b', 'dg_b', 'dg_g']
    ys, y = {}, 0.0
    for k in rows:
        if k is None:
            y += 0.7
            continue
        ys[k] = y
        y += 1
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.0), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    protos = [('best_mean1', '最佳 epoch（best checkpoint，目前的選法）'),
              ('last3_mean1', '最後 3 個 epoch 平均（ep14～16，不挑 epoch）')]
    for ax, (m, title) in zip(axes, protos):
        base = np.mean([runs[('long', s)][m] for s in seeds_of(runs, 'long')])
        ax.axvline(base, color=C_REF, lw=1.2, zorder=1)
        ax.annotate('純長視窗平均', (base, -0.75), xytext=(4, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=8, color=MUTED, annotation_clip=False)
        for k, yy in ys.items():
            v = np.array([runs[(k, s)][m] for s in seeds_of(runs, k)])
            ax.scatter(v, np.full(len(v), yy), s=34, facecolors=SURF,
                       edgecolors=C_MAIN, linewidths=1.4, zorder=3)
            ax.plot([v.mean()] * 2, [yy - 0.3, yy + 0.3], color=INK, lw=2.2, zorder=4,
                    solid_capstyle='butt')
        ax.xaxis.set_major_formatter(pct)
        ax.xaxis.set_major_locator(MultipleLocator(0.01))
        ax.set_title(title)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('驗證準確率 mean1')
    axes[0].set_yticks(list(ys.values()))
    axes[0].set_yticklabels([f'{LABEL[k]}（n={len(seeds_of(runs, k))}）' for k in ys])
    axes[0].set_ylim(max(ys.values()) + 0.8, -1.2)
    axes[0].tick_params(axis='y', length=0)
    handles = [Line2D([], [], marker='o', ls='', markerfacecolor=SURF,
                      markeredgecolor=C_MAIN, markeredgewidth=1.4, markersize=7),
               Line2D([], [], color=INK, lw=2.2), Line2D([], [], color=C_REF, lw=1.2)]
    fig.suptitle('全部設定一覽：每個點是一個 seed，黑色短線是平均',
                 x=0.01, y=1.03, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    legend_top(fig, handles, ['單一 seed', 'seed 平均', '純長視窗（基準）的平均'], y=0.985)
    fig.text(0.01, -0.03, '同一個設定的 3 個點彼此之間的距離，就是「換個 seed 會差多少」。'
             '設定之間的差距若比這個小，就分不出高下。', fontsize=8.5, color=MUTED)
    fig.subplots_adjust(top=0.87)
    fig.savefig(fname)
    plt.close(fig)


def fig_forest(Q, fname):
    heads = [('lower', '下半身 6 類（macro recall）'), ('upper', '上半身 2 類（macro recall）')]
    n = len(Q)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 0.78 * n + 1.9), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    ys = np.arange(n)
    for ax, (h, title) in zip(axes, heads):
        ax.axvline(0, color=INK2, lw=1.0, zorder=1)
        for y, q in zip(ys, Q):
            r = q['res'][h]
            col = C_MAIN if r['sig'] else C_REF
            ax.plot([r['lo'] * 100, r['hi'] * 100], [y, y], color=col, lw=2.2, zorder=2)
            k = len(r['per_unit'])
            ax.scatter(r['per_unit'] * 100, np.full(k, y + 0.26), s=16, facecolors=SURF,
                       edgecolors=col, linewidths=1.0, zorder=3)
            ax.scatter([r['delta'] * 100], [y], s=70, color=col, edgecolors=SURF,
                       linewidths=2.0, zorder=4)
        ax.set_title(title)
        ax.set_xlabel('差異（百分點；正值 = 比對照組好）')
        ax.xaxis.set_major_formatter(pp)
        ax.grid(axis='y', visible=False)
    vals = [v for q in Q for h, _ in heads
            for v in (q['res'][h]['lo'] * 100, q['res'][h]['hi'] * 100,
                      *(q['res'][h]['per_unit'] * 100))]
    for ax in axes:                                   # 兩欄同一個尺度，大小才能直接比
        ax.set_xlim(min(vals) - 0.8, max(max(vals), 1.0) + 0.8)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f'{q["title"]}\n{q["formula"]}' for q in Q], fontsize=9)
    axes[0].set_ylim(n - 0.4, -0.7)
    axes[0].tick_params(axis='y', length=0)
    handles = [Line2D([], [], marker='o', ls='', color=C_MAIN, markersize=8),
               Line2D([], [], marker='o', ls='', color=C_REF, markersize=8),
               Line2D([], [], color=INK2, lw=2.2),
               Line2D([], [], marker='o', ls='', markerfacecolor=SURF,
                      markeredgecolor=INK2, markersize=5)]
    fig.suptitle('逐題回答：這個設定有沒有用？（同 seed、同一批驗證片段相減）',
                 x=0.01, y=1.02, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    legend_top(fig, handles, ['信賴區間不含 0 → 有顯著差異', '信賴區間含 0 → 分不出來',
                              '95% 信賴區間', '各 seed 的差異'], y=0.975)
    fig.text(0.01, -0.02, '信賴區間 = 兩層 bootstrap（同時重抽驗證 clip 與 seed，4000 次）。'
             '用的是各 run 的 best checkpoint（VAL/diag_*.npz 的逐視窗預測）。',
             fontsize=8.5, color=MUTED)
    fig.subplots_adjust(top=0.86)
    fig.savefig(fname)
    plt.close(fig)


def fig_counterfactual(abl, fname):
    keys = [k for k in ('pg', 'dg_g') if k in abl] + \
           [k for k in abl if k not in ('pg', 'dg_g')]
    rows = [a for a in ABL_NAMES if any(a in d['ablation'] for k in keys
                                        for d in abl[k].values())]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 0.42 * len(rows) + 1.9),
                             sharey=True, gridspec_kw=dict(wspace=0.06))
    axes = np.atleast_1d(axes)
    ys = np.arange(len(rows))
    xmax = 0
    for ax, k in zip(axes, keys):
        ax.axvline(0, color=INK2, lw=1.0, zorder=1)
        for y, a in zip(ys, rows):
            ds = [d for _, d in sorted(abl[k].items()) if a in d['ablation']]
            if not ds:
                ax.text(0.3, y, '不適用（此設定沒有這個成分）', va='center', fontsize=8,
                        color=MUTED)
                continue
            v = np.array([(d['ablation']['normal']['macro'] - d['ablation'][a]['macro']) * 100
                          for d in ds])
            col = C_MAIN if (v > 0).all() and v.mean() >= 1.0 else C_REF
            ax.plot([v.min(), v.max()], [y, y], color=col, lw=1.4, zorder=2)
            ax.scatter(v, np.full(len(v), y), s=16, facecolors=SURF, edgecolors=col,
                       linewidths=1.0, zorder=3)
            ax.scatter([v.mean()], [y], s=60, color=col, edgecolors=SURF, linewidths=2,
                       zorder=4)
            if v.mean() >= 1.0:
                ax.annotate(f'{v.mean():.1f}', (v.max(), y), xytext=(7, 0),
                            textcoords='offset points', va='center', fontsize=8.5, color=INK2)
            xmax = max(xmax, v.max())
        ax.set_title(f'{LABEL[k]}（n={len(abl[k])} seed）')
        ax.set_xlabel('推論時關掉後，下半身 macro recall 掉了幾個百分點')
        ax.grid(axis='y', visible=False)
    for ax in axes:
        ax.set_xlim(-1.5, xmax * 1.12 + 0.5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([ABL_NAMES[a] for a in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.6)
    axes[0].tick_params(axis='y', length=0)
    handles = [Line2D([], [], marker='o', ls='', color=C_MAIN, markersize=8),
               Line2D([], [], marker='o', ls='', color=C_REF, markersize=8),
               Line2D([], [], marker='o', ls='', markerfacecolor=SURF,
                      markeredgecolor=INK2, markersize=5)]
    fig.suptitle('反事實消融：同一個訓練好的模型，推論時把某個成分關掉',
                 x=0.01, y=1.03, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    legend_top(fig, handles, ['3 個 seed 都掉、平均掉 ≥ 1 點 → 模型明顯依賴它',
                              '其餘', '各 seed'], y=0.985)
    fig.text(0.01, -0.02, '注意：這只回答「模型有沒有在用它」，不回答「加了它有沒有比較好」。'
             '後者要看重新訓練的比較（fig2）。', fontsize=8.5, color=MUTED)
    fig.subplots_adjust(top=0.84)
    fig.savefig(fname)
    plt.close(fig)


def fig_reliance(diag, abl, fname):
    """同一個問題的兩種答法並排：推論時關掉 vs 訓練時就沒有"""
    panels = []
    if 'pg' in abl:
        ci = LOWER_NAMES.index('長刺')
        rows = [
            ('單門控（正常）', [d['ablation']['normal']['recall'][ci] for d in abl['pg'].values()], C_REF),
            ('單門控，推論時拿掉短視窗', [d['ablation']['no_FS']['recall'][ci] for d in abl['pg'].values()], C_ALT),
            ('純長視窗（訓練時就沒有短視窗）', [diag[('long', s)]['lower']['recall'][ci]
                                        for s in (42, 43, 44) if ('long', s) in diag], C_MAIN),
        ]
        panels.append(('短視窗：「長刺」類別的 recall', rows, (0, 1.0)))
    if 'dg_g' in abl and any('no_inject' in d['ablation'] for d in abl['dg_g'].values()):
        rows = [
            ('雙門控+γ（正常）', [d['ablation']['normal']['macro'] for d in abl['dg_g'].values()], C_REF),
            ('雙門控+γ，推論時關掉注入', [d['ablation']['no_inject']['macro'] for d in abl['dg_g'].values()], C_ALT),
            ('雙門控（訓練時就沒有注入）', [diag[('dg', s)]['overall']['lower_mean1']
                                     for s in (42, 43, 44) if ('dg', s) in diag], C_MAIN),
        ]
        panels.append(('運動參數 γ：下半身 macro recall', rows, (0.74, 0.83)))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 2.9),
                             gridspec_kw=dict(wspace=0.75))
    axes = np.atleast_1d(axes)
    for ax, (title, rows, xl) in zip(axes, panels):
        for y, (lab, v, col) in enumerate(rows):
            v = np.asarray(v)
            ax.scatter(v, np.full(len(v), y), s=22, facecolors=SURF, edgecolors=col,
                       linewidths=1.2, zorder=3)
            ax.scatter([v.mean()], [y], s=80, color=col, edgecolors=SURF, linewidths=2,
                       zorder=4)
            ax.annotate(f'{v.mean() * 100:.1f}%', (v.mean(), y), xytext=(0, 9),
                        textcoords='offset points', ha='center', fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(len(rows) - 0.4, -0.7)
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(*xl)
        ax.xaxis.set_major_formatter(pct)
        ax.grid(axis='y', visible=False)
        ax.set_title(title)
    fig.suptitle('「推論時關掉會掉很多」≠「訓練時加上它比較好」',
                 x=0.01, y=1.08, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    fig.text(0.01, -0.08, '橘 = 同一個模型推論時硬拔掉（模型已經把工作交給這個成分了，所以崩）；'
             '藍 = 從頭訓練一個沒有這個成分的模型（它會用別的特徵學回來）。大點 = 平均，小圈 = 各 seed。',
             fontsize=8.5, color=MUTED)
    fig.savefig(fname)
    plt.close(fig)


def fig_all_settings(runs, metric, fname):
    keys = [k for k in ORDER if seeds_of(runs, k)]
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.5, 2.9 * nrow + 0.6), sharex=True,
                             sharey=True)
    ref = stack(runs, 'long', metric).mean(0) if seeds_of(runs, 'long') else None
    for ax, k in zip(axes.flat, keys):
        x = runs[(k, seeds_of(runs, k)[0])]['epoch']
        if ref is not None and k != 'long':
            ax.plot(x, ref, color=C_REF, lw=1.3, zorder=2)
        band_line(ax, x, stack(runs, k, metric), C_MAIN, LABEL[k])
        ax.set_title(f'{LABEL[k]}（n={len(seeds_of(runs, k))}）', fontsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        if metric != 'loss':
            pct_axis(ax.yaxis)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    what = '驗證準確率 mean1' if metric == 'mean1' else '訓練 loss（每個 epoch 平均）'
    fig.suptitle(f'每個設定的{what}', x=0.01, y=1.02, ha='left', fontsize=12.5,
                 fontweight='bold', color=INK)
    legend_top(fig, [Line2D([], [], color=C_MAIN, lw=2.4), Line2D([], [], color=C_REF, lw=1.3)],
               ['該設定（seed 平均，陰影 = 最高～最低）', '純長視窗（基準）的平均'], y=0.985)
    fig.subplots_adjust(top=0.9, hspace=0.35, wspace=0.08)
    fig.savefig(fname)
    plt.close(fig)


def fig_per_setting(runs, key, fname):
    ss = seeds_of(runs, key)
    x = runs[(key, ss[0])]['epoch']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    m = band_line(axes[0], x, stack(runs, key, 'mean1'), C_MAIN, '整體 mean1')
    band_line(axes[0], x, stack(runs, key, 'lower'), C_ALT, '下半身 mean1')
    band_line(axes[0], x, stack(runs, key, 'upper'), C_AQUA, '上半身 mean1')
    b = int(np.argmax(m))
    axes[0].scatter([x[b]], [m[b]], s=46, color=C_MAIN, edgecolors=SURF, linewidths=2, zorder=5)
    axes[0].annotate(f'平均曲線最高：ep{x[b]}，{m[b] * 100:.1f}%', (x[b], m[b]), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontsize=8.5, color=INK2)
    acc_axis(axes[0], '驗證準確率')
    band_line(axes[1], x, stack(runs, key, 'loss'), C_MAIN, '訓練 loss')
    loss_axis(axes[1])
    handles = [Line2D([], [], color=c, lw=2.4) for c in (C_MAIN, C_ALT, C_AQUA)]
    src = '' if FOLDER[key] == LABEL[key] else f'{FOLDER[key]}，'
    fig.suptitle(f'{LABEL[key]}（{src}n={len(ss)}：seed {", ".join(map(str, ss))}）',
                 x=0.01, y=1.05, ha='left', fontsize=12.5, fontweight='bold', color=INK)
    legend_top(fig, handles, ['整體 mean1（上下半身平均）', '下半身 6 類', '上半身 2 類'],
               y=0.995)
    fig.text(0.01, -0.04, '實線 = seed 平均；陰影 = 各 seed 的最高～最低。右圖只有一條線：'
             '訓練總 loss（上半身 loss×0.5 + 下半身 loss）。', fontsize=8.5, color=MUTED)
    fig.subplots_adjust(top=0.84, wspace=0.22)
    fig.savefig(fname)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  5. 表格
# ══════════════════════════════════════════════════════════════════
def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def f4(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.4f}'


def main(a):
    setup_style()
    out = a.out
    os.makedirs(os.path.join(out, 'per_setting'), exist_ok=True)

    runs = load_logs(a.work)
    diag, abl = load_val(a.val)
    print(f'讀到 {len(runs)} 個訓練 run、{len(diag)} 份 diag、'
          f'{sum(len(v) for v in abl.values())} 份反事實消融')
    for k in ORDER:
        s_log, s_diag = seeds_of(runs, k), sorted(s for kk, s in diag if kk == k)
        miss = sorted(set(s_log) - set(s_diag))
        if miss:
            print(f'[注意] {LABEL[k]}：seed {miss} 有訓練 log 但 VAL 沒有 diag，配對分析會少這幾組')

    # ── runs.csv（含 diag 與 log 的一致性檢查）────────────────
    rows = []
    for k in ORDER:
        for s in seeds_of(runs, k):
            r, d = runs[(k, s)], diag.get((k, s))
            dm = d['overall']['mean1'] if d else None
            rows.append([LABEL[k], FOLDER[k], s, r['best_epoch'], f4(r['best_mean1']),
                         f4(r['best_lower']), f4(r['best_upper']), f4(r['last3_mean1']),
                         f4(r['last3_lower']), f4(r['last3_upper']), f4(float(r['mean1'][-1])),
                         f4(float(r['loss'][-1])), f4(dm),
                         '' if dm is None else ('OK' if abs(dm - r['best_mean1']) < 2e-3 else '不一致'),
                         os.path.relpath(r['path'], ROOT)])
    write_csv(os.path.join(out, 'runs.csv'),
              ['設定', '資料夾', 'seed', '最佳epoch', '最佳mean1', '最佳時下半身', '最佳時上半身',
               '最後3ep_mean1', '最後3ep_下半身', '最後3ep_上半身', 'ep16_mean1', 'ep16_訓練loss',
               'diag_mean1', 'diag與log一致', 'log路徑'], rows)
    bad = [r for r in rows if r[13] == '不一致']
    print(f'[檢查] diag 與訓練 log 的 best mean1：{len(rows) - len(bad)} 份一致'
          + (f'，{len(bad)} 份不一致：{[(r[0], r[2]) for r in bad]}' if bad else ''))

    # ── settings.csv ─────────────────────────────────────────
    def ms(v):
        v = np.asarray(v, float)
        return (f'{v.mean():.4f}', f'{v.std(ddof=1):.4f}' if len(v) > 1 else '')

    rows = []
    for k in ORDER:
        ss = seeds_of(runs, k)
        if not ss:
            continue
        g = lambda m: [runs[(k, s)][m] for s in ss]
        rec = [diag[(k, s)]['lower']['recall'] for s in ss if (k, s) in diag]
        rec = np.mean(rec, 0) if rec else [np.nan] * 6
        rows.append([LABEL[k], len(ss), *ms(g('best_mean1')), *ms(g('best_lower')),
                     *ms(g('best_upper')), *ms(g('last3_mean1')),
                     ' / '.join(str(runs[(k, s)]['best_epoch']) for s in ss),
                     *[f4(float(x)) for x in rec]])
    write_csv(os.path.join(out, 'settings.csv'),
              ['設定', 'seed數', '最佳mean1_平均', '最佳mean1_標準差', '最佳時下半身_平均',
               '最佳時下半身_標準差', '最佳時上半身_平均', '最佳時上半身_標準差',
               '最後3ep_mean1_平均', '最後3ep_mean1_標準差', '各seed最佳epoch',
               *[f'下半身recall_{n}' for n in LOWER_NAMES]], rows)

    # ── 配對比較 ─────────────────────────────────────────────
    rng = np.random.default_rng(a.seed)
    ref = next(d['_z'] for d in diag.values() if d['_z'] is not None)
    boot = Boot(ref['cid'], a.boot, rng)
    Q, rows = [], []
    for title, pairs, formula in QUESTIONS + EXTRA:
        r = compare(pairs, diag, runs, boot, rng)
        if r is None:
            print(f'[略過] {title}：缺資料')
            continue
        item = dict(title=title, formula=formula, res=r)
        if (title, pairs, formula) in QUESTIONS:
            Q.append(item)
        cls = []
        for t, c, s in r['units']:
            cls.append(np.array(diag[(t, s)]['lower']['recall'])
                       - np.array(diag[(c, s)]['lower']['recall']))
        cls = np.mean(cls, 0)
        rows.append([title, formula, len(r['units']),
                     *[x for h in ('mean1', 'lower', 'upper') for x in
                       (f'{r[h]["delta"] * 100:+.2f}', f'{r[h]["lo"] * 100:+.2f}',
                        f'{r[h]["hi"] * 100:+.2f}', f'{r[h]["agree"]}/{len(r["units"])}')],
                     ' '.join(f'{v * 100:+.2f}' for v in r['lower']['per_unit']),
                     f'{r["last3_mean1"]["delta"] * 100:+.2f}',
                     verdict(r, 'lower'), verdict(r, 'upper'),
                     *[f'{v * 100:+.1f}' for v in cls]])
    write_csv(os.path.join(out, 'comparisons.csv'),
              ['問題', '算式', '配對數',
               'Δmean1(pp)', 'mean1_CI下', 'mean1_CI上', 'mean1_同號seed',
               'Δ下半身(pp)', '下半身_CI下', '下半身_CI上', '下半身_同號seed',
               'Δ上半身(pp)', '上半身_CI下', '上半身_CI上', '上半身_同號seed',
               '下半身各配對Δ(pp)', 'Δmean1_最後3ep(pp)', '下半身判定', '上半身判定',
               *[f'Δrecall_{n}(pp)' for n in LOWER_NAMES]], rows)

    print('\n【逐題結論】（下半身 macro recall，best checkpoint，同 seed 配對）')
    for q in Q:
        x = q['res']['lower']
        print(f'  {q["title"]:<16} Δ={x["delta"] * 100:+5.2f} pp  '
              f'95%CI [{x["lo"] * 100:+5.2f}, {x["hi"] * 100:+5.2f}]  '
              f'{x["agree"]}/{len(x["per_unit"])} 同號  → {verdict(q["res"])}')

    # ── ablations.csv ────────────────────────────────────────
    rows = []
    for k, per_seed in abl.items():
        for a_ in ABL_NAMES:
            ds = [d for _, d in sorted(per_seed.items()) if a_ in d['ablation']]
            if not ds:
                continue
            dl = np.array([d['ablation']['normal']['macro'] - d['ablation'][a_]['macro'] for d in ds])
            du = np.array([d['ablation']['normal']['upper_macro'] - d['ablation'][a_]['upper_macro']
                           for d in ds])
            rc = np.mean([np.array(d['ablation']['normal']['recall'])
                          - np.array(d['ablation'][a_]['recall']) for d in ds], 0)
            rows.append([LABEL[k], a_, ABL_NAMES[a_], len(ds), f'{dl.mean() * 100:+.2f}',
                         f'{dl.std(ddof=1) * 100:.2f}' if len(dl) > 1 else '',
                         ' '.join(f'{v * 100:+.2f}' for v in dl), f'{du.mean() * 100:+.2f}',
                         *[f'{v * 100:+.1f}' for v in rc]])
    write_csv(os.path.join(out, 'ablations.csv'),
              ['設定', '消融', '說明', 'seed數', '下半身掉幾點(pp)', '標準差', '各seed',
               '上半身掉幾點(pp)', *[f'recall掉幾點_{n}' for n in LOWER_NAMES]], rows)

    # ── 圖 ───────────────────────────────────────────────────
    fig_overview(runs, os.path.join(out, 'fig1_overview.png'))
    fig_forest(Q, os.path.join(out, 'fig2_forest.png'))
    fig_compare(runs, [('long', C_REF), ('dw', C_MAIN), ('short', C_ALT)],
                '長視窗、短視窗、兩者都用：哪個好？', os.path.join(out, 'fig3a_window.png'))
    fig_compare(runs, [('dw', C_REF), ('learn', C_ALT), ('pg', C_MAIN)],
                '雙視窗的融合權重怎麼決定：固定 0.5、學一組固定權重、還是看運動特徵決定？',
                os.path.join(out, 'fig3b_fusion.png'),
                note='三者都有長短兩個視窗，只差在融合權重。')
    fig_compare(runs, [('pg', C_REF), ('dg', C_MAIN)],
                '上下半身共用一個門控 vs 各用一個門控',
                os.path.join(out, 'fig3c_gate.png'))
    fig_motion(runs, os.path.join(out, 'fig3d_motion.png'))
    if abl:
        fig_counterfactual(abl, os.path.join(out, 'fig4_counterfactual.png'))
        fig_reliance(diag, abl, os.path.join(out, 'fig5_reliance.png'))
    fig_all_settings(runs, 'mean1', os.path.join(out, 'all_settings_acc.png'))
    fig_all_settings(runs, 'loss', os.path.join(out, 'all_settings_loss.png'))
    for k in ORDER:
        if seeds_of(runs, k):
            fig_per_setting(runs, k, os.path.join(out, 'per_setting', f'{FOLDER[k]}.png'))
    print(f'\n[完成] 圖表與表格都在 {os.path.relpath(out, ROOT)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default=os.path.join(ROOT, 'work_dirs'))
    ap.add_argument('--val', default=HERE)
    ap.add_argument('--out', default=os.path.join(HERE, 'report'))
    ap.add_argument('--boot', type=int, default=4000, help='bootstrap 次數')
    ap.add_argument('--seed', type=int, default=0, help='bootstrap 亂數種子')
    main(ap.parse_args())
