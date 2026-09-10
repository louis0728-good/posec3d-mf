# -*- coding: utf-8 -*-
"""
tt2.py — 雙視窗 / 門控 / 運動參數注入 的逐類別診斷與消融

與舊版的差異
────────────────────────────────────────────────────────────────────
舊版對應的是「sg_gamma 直達路徑 + gating_machine 吃 F_L/F_S/sg」那一代 neck，
和現在的 DualWindowGatingNeck 已經對不上（見下表），因此整份重寫：

  舊版假設                          現在的 neck
  neck.use_fs / use_sg              use_lw / use_sw / use_gate / dual_gate
  neck.sg_norm / sg_proj / sg_gamma 不存在（沒有 S-G 直達加性路徑）
  neck.gating_machine(F_L,F_S,sg)   _gate_fuse(scorer(motion)) → 只吃 motion
  data_sample.sg_features           motion_L/S, motion_upper_L/S, motion_lower_L/S
  neck 回傳單一張量                  回傳 (F_up, F_low)
  單視窗維度 170 = 17*(2+8)          36（分部位）/ 72（全身）= 6 區塊 × V 關節

另一個結構性改動：舊版在 hook 裡「重新實作一次 neck.forward」，neck 一改就會
靜默算錯。新版改成只攔四個穩定介面做反事實介入，不複製 forward 邏輯：
    neck.small_proj          (forward hook)   → F_S 消融
    MotionInject.forward     (patch)          → 運動參數注入消融
    neck._gate_fuse          (patch)          → gating 消融
    neck._stack_motion       (patch)          → 運動特徵分區塊消融

主要產出（這才是重點）
────────────────────────────────────────────────────────────────────
  1. 逐類別 recall / precision / support（上半身 2 類、下半身 6 類）
  2. 混淆矩陣
  3. macro(mean1) vs micro(top1) 分解 —— 直接看出「贏在多數類還是少數類」
  4. window 級與 clip 級兩種統計
  5. cluster bootstrap 95% CI（以 frame_dir 重抽樣）
  6. 反事實消融（可選，--no-ablation 可關）
  7. --compare：跨設定 × 跨 seed 彙整，含「同一個 window 配對」的逐類別
     Δrecall 與 McNemar 檢定 —— 這比比較 3 個 seed 的平均值敏感得多

用法
────────────────────────────────────────────────────────────────────
  # 單一 checkpoint（mode / inject / scope 會自動從權重推斷）
  python tt2.py --ckpt work_dirs/.../best_dual_acc_mean1_epoch_13.pth \
                --tag dualgate_gamma_s42

  # 只要逐類別結果、不跑消融（快，約 15 秒）
  python tt2.py --ckpt ... --tag longonly_s42 --no-ablation

  # 全部跑完後彙整（--baseline 指定當基準的那組設定）
  python tt2.py --compare diag_*.json --baseline long_only

建議的最小跑法（回答「哪些類別贏、哪些輸」）
    5 個設定 × 3 個 seed = 15 次 --no-ablation，再一次 --compare
"""

import argparse
import collections
import json
import os
import sys
import glob 
import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmaction.registry import MODELS

# mmengine 的 checkpoint 需要完整反序列化
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

CFG = 'configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'

NAMES_L = ['無', '前進', '後退', '長刺', '飛刺', '前進長刺']
NAMES_U = ['上_0', '上_1']

# SG_filter.py 的 feats tuple 順序（每個區塊寬度 = V 個關節）
SG_BLOCKS = ['sp_mean', 'sp_std', 'sp_max', 'am_mean', 'am_max', 'straight']


# ══════════════════════════════════════════════════════════════════
#  0. 從 checkpoint 反推 neck 設定
# ══════════════════════════════════════════════════════════════════
def infer_neck_cfg(sd):
    """從 state_dict 的 key 推斷 mode / motion_inject / inject_scope。

    避免「拿 A 設定的 config 去載 B 設定的權重」這種靜默錯誤——
    load_checkpoint 對缺漏的 key 只會 warning，不會擋下來。
    """
    keys = list(sd.keys())
    has = lambda s: any(s in k for k in keys)

    if has('neck.motion_scorer_upper'):
        mode = 'dual_gate'
    elif has('neck.motion_scorer.'):
        mode = 'per_gate'
    elif has('neck.alpha_logit'):
        mode = 'learnable'
    elif has('neck.small_proj'):
        mode = 'dual_window'      # 有短視窗投影但沒有 gate
    else:
        mode = 'long_only'

    if has('neck.inject_upper') or has('neck.inject_lower'):
        scope = 'upper_lower'
        pre = 'neck.inject_upper'
    elif has('neck.inject_whole'):
        scope = 'whole_body'
        pre = 'neck.inject_whole'
    else:
        return mode, 'none', 'whole_body'

    to_b = has(f'{pre}.to_beta')
    to_g = has(f'{pre}.to_gamma')
    inject = 'film' if (to_b and to_g) else ('beta' if to_b else 'gamma')

    # dual_window 若同時有 inject，仍可能是 short_only；用 small_proj 的有無再確認
    if mode == 'dual_window' and not has('neck.small_proj'):
        mode = 'long_only'
    return mode, inject, scope


def sg_layout(dim):
    """回傳 {區塊名: (start, end)}。dim = 6 區塊 × V 關節。"""
    assert dim % len(SG_BLOCKS) == 0, \
        f'運動特徵維度 {dim} 不是 {len(SG_BLOCKS)} 的倍數，SG_filter 的 feats 有改過？'
    v = dim // len(SG_BLOCKS)
    out, p = {}, 0
    for b in SG_BLOCKS:
        out[b] = (p, p + v)
        p += v
    return out


# ══════════════════════════════════════════════════════════════════
#  1. 消融設定
# ══════════════════════════════════════════════════════════════════
class Ab:
    """
    fs_mean      : F_S(small_proj 輸出) → 資料集平均（等同拿掉短視窗分支，
                   但不是歸零，避免混入「量值消失」的假效應）
    motion_keys  : {key: 'ALL' 或 [(a,b), ...]} 要換成資料集平均的運動特徵區間
    inject       : None 不動 / 'off' 完全不注入 / 'L_off' 只有長視窗不注入 /
                   'S_off' 只有短視窗不注入
    gate         : None 不動 / 'half' 強制 0.5,0.5 / 'mean' 用全域逐通道平均 alpha
    need         : 需要的能力，不滿足就跳過
    """

    def __init__(self, name, desc, fs_mean=False, motion_keys=None,
                 inject=None, gate=None, need=()):
        self.name, self.desc = name, desc
        self.fs_mean = fs_mean
        self.motion_keys = motion_keys or {}
        self.inject = inject
        self.gate = gate
        self.need = need


def build_ablations(neck, motion_keys_used):
    part = neck.motion_dim_part
    whole = neck.motion_dim_whole
    A = [Ab('normal', '原始（對照組，必須復現訓練 log 的數字）')]

    A.append(Ab('no_FS', 'F_S → 資料集平均（拿掉短視窗分支）',
                fs_mean=True, need=('sw',)))

    if neck.motion_inject != 'none':
        A += [
            Ab('no_inject', '完全不注入運動參數（等同 motion_inject=none）',
               inject='off', need=('inject',)),
            Ab('inject_L_off', '只有長視窗不注入運動參數',
               inject='L_off', need=('inject', 'lw', 'sw')),
            Ab('inject_S_off', '只有短視窗不注入運動參數',
               inject='S_off', need=('inject', 'lw', 'sw')),
        ]

    # 運動特徵本身的消融（同時影響 gating 與注入）
    all_mean = {k: 'ALL' for k in motion_keys_used}
    A.append(Ab('no_motion', '全部運動特徵 → 資料集平均',
                motion_keys=all_mean, need=('motion',)))

    long_keys = {k: 'ALL' for k in motion_keys_used if k.endswith('_L')}
    short_keys = {k: 'ALL' for k in motion_keys_used if k.endswith('_S')}
    if long_keys and short_keys:
        A += [
            Ab('motion_L_off', '只拿掉長視窗(21幀)的運動特徵',
               motion_keys=long_keys, need=('motion',)),
            Ab('motion_S_off', '只拿掉短視窗(15幀)的運動特徵',
               motion_keys=short_keys, need=('motion',)),
        ]

    # 分區塊：速度類 vs 加速度類 vs 直線度
    def blocks(names):
        out = {}
        for k in motion_keys_used:
            dim = part if ('upper' in k or 'lower' in k) else whole
            lay = sg_layout(dim)
            out[k] = [lay[b] for b in names]
        return out

    A += [
        Ab('sg_acc_off', '拿掉加速度統計 (am_mean, am_max)',
           motion_keys=blocks(['am_mean', 'am_max']), need=('motion',)),
        Ab('sg_speed_off', '拿掉速度統計 (sp_mean, sp_std, sp_max)',
           motion_keys=blocks(['sp_mean', 'sp_std', 'sp_max']), need=('motion',)),
        Ab('sg_straight_off', '拿掉直線度 (straight)',
           motion_keys=blocks(['straight']), need=('motion',)),
    ]

    A += [
        Ab('gate_half', 'alpha=beta=0.5（退化成等權相加）',
           gate='half', need=('gate',)),
        Ab('gate_mean', 'alpha → 全域逐通道平均（保留通道分工，去掉逐樣本自適應）',
           gate='mean', need=('gate',)),
    ]
    return A


# ══════════════════════════════════════════════════════════════════
#  2. 介入（hook / patch），不重新實作 forward
# ══════════════════════════════════════════════════════════════════
class Intervene:
    """以 context manager 方式套用消融，離開時保證還原。"""

    def __init__(self, model, ab, base, rec=None):
        self.model, self.ab, self.base = model, ab, base
        self.rec = rec              # 不是 None 就順便記錄內部量
        self.handles, self.restore = [], []

    def __enter__(self):
        neck = self.model.neck
        ab, base, rec = self.ab, self.base, self.rec

        # ── (a) small_proj：記錄 F_S / 換成平均 ────────────────
        if getattr(neck, 'use_sw', False) and hasattr(neck, 'small_proj'):
            def fs_hook(mod, inp, out):
                if rec is not None:
                    rec['_FS'].append(out.detach().float().cpu().numpy())
                    rec['n_FS'].append(
                        out.detach().float().norm(dim=1).cpu().numpy())
                if ab.fs_mean:
                    m = base['FS'].to(out.device).to(out.dtype)
                    return m.unsqueeze(0).expand(out.shape[0], -1).contiguous()
                return None
            self.handles.append(neck.small_proj.register_forward_hook(fs_hook))

        # ── (b) _stack_motion：記錄 / 分區塊換平均 ─────────────
        orig_stack = neck._stack_motion

        def stack_patch(data_samples, key, batch_size, device, expect_dim):
            m = orig_stack(data_samples, key, batch_size, device, expect_dim)
            if rec is not None:
                rec['_M_' + key].append(m.detach().float().cpu().numpy())
            spec = ab.motion_keys.get(key)
            if spec is not None:
                mu = base['M'][key].to(device).to(m.dtype)
                m = m.clone()
                spans = [(0, m.shape[1])] if spec == 'ALL' else spec
                for a0, a1 in spans:
                    m[:, a0:a1] = mu[a0:a1]
            return m

        neck._stack_motion = stack_patch
        self.restore.append(('_stack_motion', orig_stack))

        # ── (c) MotionInject.forward：關掉注入 ─────────────────
        if ab.inject is not None and neck.motion_inject != 'none':
            slots = (['L'] if neck.use_lw else []) + (['S'] if neck.use_sw else [])
            targets = [getattr(neck, n) for n in
                       ('inject_upper', 'inject_lower', 'inject_whole')
                       if hasattr(neck, n)]
            for inj in targets:
                orig_fwd = inj.forward

                def make(orig_fwd=orig_fwd):
                    def patched(feat_list, motion_list):
                        if ab.inject == 'off':
                            return list(feat_list)
                        out = orig_fwd(feat_list, motion_list)
                        keep = 'L' if ab.inject == 'L_off' else 'S'
                        return [feat_list[i] if slots[i] == keep else out[i]
                                for i in range(len(out))]
                    return patched

                inj.forward = make()
                self.restore.append((inj, orig_fwd))

        # ── (d) _gate_fuse：gating 消融 + 記錄 alpha ───────────
        if getattr(neck, 'use_gate', False):
            orig_gate = neck._gate_fuse

            def gate_patch(scorer, motion_L, motion_S, F_L, F_S, tag=''):
                pair = torch.cat([motion_L, motion_S], dim=0)
                s_pair = scorer(pair)
                sL, sS = s_pair.chunk(2, dim=0)
                w = torch.softmax(torch.stack([sL, sS], dim=1), dim=1)
                alpha, beta = w[:, 0], w[:, 1]

                key = tag.strip() or 'all'
                if rec is not None:
                    rec['alpha_' + key].append(
                        alpha.detach().float().cpu().numpy())

                if ab.gate == 'half':
                    alpha = torch.full_like(alpha, 0.5)
                    beta = torch.full_like(beta, 0.5)
                elif ab.gate == 'mean':
                    a = base['ALPHA'][key].to(alpha.device).to(alpha.dtype)
                    alpha = a.unsqueeze(0).expand_as(alpha).contiguous()
                    beta = 1.0 - alpha
                return alpha * F_L + beta * F_S

            neck._gate_fuse = gate_patch
            self.restore.append(('_gate_fuse', orig_gate))

        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        for tgt, fn in self.restore:
            if isinstance(tgt, str):
                setattr(self.model.neck, tgt, fn)
            else:
                tgt.forward = fn
        return False


# ══════════════════════════════════════════════════════════════════
#  3. 推論與收集
# ══════════════════════════════════════════════════════════════════
_KEY = {}


def pick_scores(s):
    if 'u' not in _KEY:
        cu = ['pred_score_upper', 'pred_scores_upper', 'pred_upper']
        cl = ['pred_score_lower', 'pred_scores_lower', 'pred_lower']
        _KEY['u'] = next((k for k in cu if hasattr(s, k)), None)
        _KEY['l'] = next((k for k in cl if hasattr(s, k)), None)
        if _KEY['u'] is None or _KEY['l'] is None:
            raise RuntimeError(
                f'找不到 upper/lower 分數屬性。可用的 key: {list(s.keys())}\n'
                f'請把正確名稱補進 pick_scores 的候選清單。')
        print(f'[info] 使用 {_KEY["u"]} / {_KEY["l"]}')
    return (getattr(s, _KEY['u']).detach().float().cpu().numpy(),
            getattr(s, _KEY['l']).detach().float().cpu().numpy())


def run(model, loader, ab, base, record=False):
    rec = collections.defaultdict(list) if record else None
    Pu, Pl, Gu, Gl, Cid = [], [], [], [], []
    with Intervene(model, ab, base, rec), torch.no_grad():
        for batch in loader:
            for s in model.val_step(batch):
                u, l = pick_scores(s)
                Pu.append(u)
                Pl.append(l)
                g = np.asarray(s.gt_label.detach().cpu()).reshape(-1)
                Gu.append(int(g[0]))
                Gl.append(int(g[1]))
                Cid.append(str(getattr(s, 'frame_dir', '?')))
    out = dict(Pu=np.array(Pu), Pl=np.array(Pl),
               Gu=np.array(Gu), Gl=np.array(Gl), Cid=Cid)
    if rec is not None:
        for k, v in rec.items():
            out[k] = np.concatenate(v, axis=0)
    return out


# ══════════════════════════════════════════════════════════════════
#  4. 指標
# ══════════════════════════════════════════════════════════════════
def per_class(P, G, n):
    """回傳 macro(mean1), micro(top1), 每類 recall / precision / support"""
    p = P.argmax(1)
    rec, prec, sup = [], [], []
    for c in range(n):
        m = (G == c)
        sup.append(int(m.sum()))
        rec.append(float((p[m] == c).mean()) if m.any() else 0.0)
        q = (p == c)
        prec.append(float((G[q] == c).mean()) if q.any() else 0.0)
    return (float(np.mean(rec)), float((p == G).mean()),
            rec, prec, sup)


def confusion(P, G, n):
    cm = np.zeros((n, n), int)
    for t, q in zip(G, P.argmax(1)):
        cm[t, q] += 1
    return cm


def by_clip(P, G, cid):
    """同一 frame_dir 的多個 window 取機率平均，得到 clip 級預測"""
    idx = collections.defaultdict(list)
    for i, c in enumerate(cid):
        idx[c].append(i)
    keys = sorted(idx)
    Pc = np.stack([P[idx[k]].mean(0) for k in keys])
    Gc = np.array([G[idx[k][0]] for k in keys])
    return Pc, Gc, keys


def clip_boot(cid, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = collections.defaultdict(list)
    for i, c in enumerate(cid):
        idx[c].append(i)
    keys = list(idx)
    return [np.concatenate([idx[keys[j]] for j in
                            rng.integers(0, len(keys), len(keys))])
            for _ in range(B)]


# ══════════════════════════════════════════════════════════════════
#  5. 主流程
# ══════════════════════════════════════════════════════════════════
def main(a):
    init_default_scope('mmaction')
    cfg = Config.fromfile(a.cfg)

    ck = torch.load(a.ckpt, map_location='cpu')
    sd = ck.get('state_dict', ck)
    mode, inject, scope = infer_neck_cfg(sd)
    if a.mode:
        mode = a.mode
    if a.inject:
        inject = a.inject
    if a.scope:
        scope = a.scope

    cfg.model.neck.mode = mode
    cfg.model.neck.motion_inject = inject
    cfg.model.neck.inject_scope = scope
    cfg.model.neck.debug = False
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    cfg.val_dataloader.sampler.shuffle = False
    if a.batch:
        cfg.val_dataloader.batch_size = a.batch

    loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    info = load_checkpoint(model, a.ckpt, map_location='cpu')
    model = model.cuda().eval()
    neck = model.neck

    nL = cfg.model.cls_head.num_classes_lower
    nU = cfg.model.cls_head.num_classes_upper
    names_l = (a.names_lower.split(',') if a.names_lower
               else NAMES_L[:nL] if nL <= len(NAMES_L) else
               [f'L{i}' for i in range(nL)])
    names_u = NAMES_U[:nU] if nU <= len(NAMES_U) else [f'U{i}' for i in range(nU)]

    cap = dict(lw=neck.use_lw, sw=neck.use_sw, gate=neck.use_gate,
               inject=neck.motion_inject != 'none',
               motion=neck.use_gate or neck.motion_inject != 'none')

    print('=' * 78)
    print(f'ckpt   = {a.ckpt}')
    print(f'推斷   = mode={mode}  motion_inject={inject}  inject_scope={scope}')
    print(f'能力   = 長視窗{cap["lw"]}  短視窗{cap["sw"]}  '
          f'門控{cap["gate"]}  注入{cap["inject"]}')
    print('=' * 78)
    print('  ↑ 若這行與你訓練時的 config 不符，後面全部無效，請用 '
          '--mode/--inject/--scope 手動指定')

    S = dict(tag=a.tag or os.path.basename(a.ckpt), ckpt=a.ckpt,
             mode=mode, motion_inject=inject, inject_scope=scope,
             note=a.note)

    # ── 第一趟：normal，順便收 baseline ───────────────────────
    print('\n[run] normal ...')
    R0 = run(model, loader, Ab('normal', ''), base={}, record=True)

    base = {}
    if '_FS' in R0:
        base['FS'] = torch.from_numpy(R0['_FS'].mean(0))
    base['M'] = {k[3:]: torch.from_numpy(v.mean(0))
                 for k, v in R0.items() if k.startswith('_M_')}
    base['ALPHA'] = {k[6:]: torch.from_numpy(v.mean(0))
                     for k, v in R0.items() if k.startswith('alpha_')}
    motion_keys_used = list(base['M'])

    mlo, tlo, rl, pl, sl_ = per_class(R0['Pl'], R0['Gl'], nL)
    muo, tuo, ru, pu, su = per_class(R0['Pu'], R0['Gu'], nU)

    print(f'\n★ sanity check')
    print(f'  lower_mean1 = {mlo:.4f}   lower_top1 = {tlo:.4f}')
    print(f'  upper_mean1 = {muo:.4f}   upper_top1 = {tuo:.4f}')
    print(f'  mean1 = {(mlo+muo)/2:.4f}   top1 = {(tlo+tuo)/2:.4f}')
    print('  ↑ mean1 必須與訓練 log 的該 epoch 一致；不一致就是 config 或 '
          'ckpt 對錯了，別往下看')

    S['overall'] = dict(lower_mean1=mlo, lower_top1=tlo,
                        upper_mean1=muo, upper_top1=tuo,
                        mean1=(mlo + muo) / 2, top1=(tlo + tuo) / 2,
                        n_window=int(len(R0['Gl'])),
                        n_clip=int(len(set(R0['Cid']))))

    # ── 逐類別 ────────────────────────────────────────────────
    print(f'\n【1. 下半身逐類別】(window 級, n={len(R0["Gl"])})')
    print(f'  {"類別":<12}{"n":>6}{"recall":>9}{"precision":>11}'
          f'{"單樣本翻轉對 macro 的影響":>26}')
    for i in range(nL):
        imp = 100.0 / max(sl_[i], 1) / nL
        print(f'  {names_l[i]:<12}{sl_[i]:6d}{rl[i]:9.4f}{pl[i]:11.4f}'
              f'{imp:22.3f} pp')
    print(f'  {"macro(mean1)":<12}{"":>6}{mlo:9.4f}')
    print(f'  {"micro(top1)":<12}{"":>6}{tlo:9.4f}')
    print(f'  → macro − micro = {mlo - tlo:+.4f}   '
          f'（負值越大代表越靠多數類撐分數）')
    print('  ↑ 最右欄是「單一樣本翻轉能移動 macro 幾個百分點」。'
          '若它接近你表上的設定間差距，那個差距就在噪音量級內')

    print(f'\n【2. 上半身逐類別】')
    for i in range(nU):
        print(f'  {names_u[i]:<12}{su[i]:6d}{ru[i]:9.4f}{pu[i]:11.4f}')
    print(f'  macro={muo:.4f}  micro={tuo:.4f}  差={muo - tuo:+.4f}')

    S['lower'] = dict(recall=rl, precision=pl, support=sl_, names=names_l)
    S['upper'] = dict(recall=ru, precision=pu, support=su, names=names_u)

    cm = confusion(R0['Pl'], R0['Gl'], nL)
    print(f'\n【3. 下半身混淆矩陣】(列=真實 行=預測)')
    print(f'{"":14}' + ''.join(f'{n:>9}' for n in names_l) + f'{"recall":>9}')
    for i in range(nL):
        print(f'{names_l[i]:<14}' + ''.join(f'{cm[i,j]:9d}' for j in range(nL))
              + f'{rl[i]:9.4f}')
    S['confusion_lower'] = cm.tolist()

    # 最常見的混淆對
    off = [(cm[i, j], i, j) for i in range(nL) for j in range(nL) if i != j]
    off.sort(reverse=True)
    print('  最常見的錯誤：' + '  '.join(
        f'{names_l[i]}→{names_l[j]}({c})' for c, i, j in off[:4]))

    # ── clip 級 ───────────────────────────────────────────────
    Pc, Gc, ckeys = by_clip(R0['Pl'], R0['Gl'], R0['Cid'])
    mlc, tlc, rlc, _, slc = per_class(Pc, Gc, nL)
    print(f'\n【4. clip 級（同 frame_dir 的 window 機率平均）】n={len(Gc)}')
    print(f'  lower macro={mlc:.4f}  micro={tlc:.4f}')
    print('  ' + '  '.join(f'{names_l[i]}={rlc[i]:.3f}' for i in range(nL)))
    S['lower_clip'] = dict(macro=mlc, micro=tlc, recall=rlc, support=slc)

    # ── gating 行為 ───────────────────────────────────────────
    if cap['gate']:
        print('\n【5. gating 的 alpha（長視窗權重）】')
        S['gate'] = {}
        for key in sorted(base['ALPHA']):
            av = R0['alpha_' + key]
            am = av.mean(0)
            print(f'  [{key}] 全體平均={av.mean():.4f}  '
                  f'跨通道 std={am.std():.4f}  '
                  f'逐樣本 std={av.mean(1).std():.4f}')
            print(f'        alpha>0.6 通道={int((am>.6).sum())}  '
                  f'<0.4 通道={int((am<.4).sum())} / {am.size}')
            S['gate'][key] = dict(mean=float(av.mean()),
                                  chan_std=float(am.std()),
                                  sample_std=float(av.mean(1).std()),
                                  n_hi=int((am > .6).sum()),
                                  n_lo=int((am < .4).sum()))
            print(f'        每類平均 alpha：' + '  '.join(
                f'{names_l[c]}={av[R0["Gl"]==c].mean():.3f}'
                for c in range(nL) if (R0['Gl'] == c).any()))
        print('  ↑ 跨通道 std 大 = 有通道分工；逐樣本 std 大 = 有樣本自適應。'
              '兩者都接近 0 代表 gating 退化成固定權重')

    # ── bootstrap CI ──────────────────────────────────────────
    picks = clip_boot(R0['Cid'], a.boot) if a.boot else None
    if picks:
        bl = np.array([per_class(R0['Pl'][i], R0['Gl'][i], nL)[0] for i in picks])
        bt = np.array([per_class(R0['Pl'][i], R0['Gl'][i], nL)[1] for i in picks])
        lo, hi = np.percentile(bl, [2.5, 97.5])
        lo2, hi2 = np.percentile(bt, [2.5, 97.5])
        print(f'\n【6. cluster bootstrap 95% CI】(以 clip 重抽樣, B={a.boot})')
        print(f'  lower macro = {mlo:.4f}  CI=[{lo:.4f}, {hi:.4f}]  '
              f'寬度={hi-lo:.4f}')
        print(f'  lower micro = {tlo:.4f}  CI=[{lo2:.4f}, {hi2:.4f}]  '
              f'寬度={hi2-lo2:.4f}')
        print('  ↑ 若寬度遠大於設定間差距，主表就必須報 multi-seed mean±std，'
              '單一 checkpoint 的數字沒有意義')
        S['ci'] = dict(macro=[float(lo), float(hi)],
                       micro=[float(lo2), float(hi2)])

    # ── 消融 ──────────────────────────────────────────────────
    if not a.no_ablation:
        for k in list(R0):
            if k.startswith('_'):
                R0.pop(k)
        abs_all = build_ablations(neck, motion_keys_used)
        print('\n【7. 反事實消融】(Δ = 原始 − 消融後，正值代表該成分有貢獻)')
        hdr = (f'{"設定":<18}{"macro":>8}{"micro":>8}{"Δmacro":>9}{"Δmicro":>9}'
               f'{"Δupper":>9}')
        print(hdr)
        print('-' * 62)
        tab = {}
        for ab in abs_all:
            if not all(cap[n] for n in ab.need):
                continue
            R = R0 if ab.name == 'normal' else run(model, loader, ab, base)
            ml, tl, r, _, _ = per_class(R['Pl'], R['Gl'], nL)
            mu, tu, _, _, _ = per_class(R['Pu'], R['Gu'], nU)
            row = f'{ab.name:<18}{ml:8.4f}{tl:8.4f}'
            if ab.name != 'normal':
                row += f'{mlo-ml:+9.4f}{tlo-tl:+9.4f}{muo-mu:+9.4f}'
            print(row)
            tab[ab.name] = dict(macro=ml, micro=tl, upper_macro=mu,
                                recall=r, desc=ab.desc)
        S['ablation'] = tab

        print('\n  逐類別 Δrecall（原始 − 消融後）')
        print(f'  {"設定":<18}' + ''.join(f'{n:>10}' for n in names_l))
        for k, v in tab.items():
            if k == 'normal':
                continue
            print(f'  {k:<18}' + ''.join(
                f'{rl[i]-v["recall"][i]:+10.4f}' for i in range(nL)))

        print('\n  說明：')
        for ab in abs_all:
            if ab.name in tab and ab.desc:
                print(f'    {ab.name:<18} {ab.desc}')

        if 'gate_half' in tab and 'gate_mean' in tab:
            print(f'\n  gating 價值分解（lower macro）：')
            print(f'    逐樣本自適應 = {mlo - tab["gate_mean"]["macro"]:+.4f}')
            print(f'    靜態通道分工 = '
                  f'{tab["gate_mean"]["macro"] - tab["gate_half"]["macro"]:+.4f}')
            print(f'    gating 總價值 = {mlo - tab["gate_half"]["macro"]:+.4f}')
            print('    ↑ gate_half 就是「等權相加」。這一行直接量出門控相對'
                  '等權到底值多少，以及來自通道分工還是樣本自適應')

        if 'no_FS' in tab and 'no_motion' in tab:
            d1 = mlo - tab['no_FS']['macro']
            d2 = mlo - tab['no_motion']['macro']
            print(f'\n  短視窗分支 Δ={d1:+.4f}   運動特徵 Δ={d2:+.4f}')
            print('    ↑ 兩者都很小代表這兩個模組在這份權重上幾乎沒被用到')

    # ── 存檔 ──────────────────────────────────────────────────
    tag = a.tag or os.path.splitext(os.path.basename(a.ckpt))[0]
    out = a.out or f'diag_{tag}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(S, f, ensure_ascii=False, indent=2)
    # 預測結果另存，--compare 的配對分析要用
    np.savez_compressed(
        os.path.splitext(out)[0] + '.npz',
        pred_lower=R0['Pl'].astype(np.float32),
        pred_upper=R0['Pu'].astype(np.float32),
        gt_lower=R0['Gl'], gt_upper=R0['Gu'],
        cid=np.array(R0['Cid']))
    print(f'\n[saved] {out}  +  {os.path.splitext(out)[0]}.npz')


# ══════════════════════════════════════════════════════════════════
#  6. 跨設定 / 跨 seed 彙整
# ══════════════════════════════════════════════════════════════════
def load_all(files):
    # cmd 不會展開萬用字元，這裡自己展開
    expanded = []
    for f in files:
        expanded.extend(sorted(glob.glob(f)) if any(c in f for c in '*?[') else [f])
    files = expanded
    out = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
        npz = os.path.splitext(f)[0] + '.npz'
        d['_npz'] = np.load(npz, allow_pickle=True) if os.path.exists(npz) else None
        d['_file'] = f
        d['_cfg'] = f'{d["mode"]}/{d["motion_inject"]}/{d["inject_scope"]}'
        out.append(d)
    return out


def ms(xs):
    xs = list(xs)
    if len(xs) < 2:
        return f'{xs[0]:.4f}' if xs else '-'
    return f'{np.mean(xs):.4f}±{np.std(xs, ddof=1):.4f}'


def compare(files, baseline):
    D = load_all(files)
    groups = collections.OrderedDict()
    for d in D:
        groups.setdefault(d['_cfg'], []).append(d)

    names = D[0]['lower']['names']
    nL = len(names)

    print('=' * 92)
    print('【A. 主表】mean±std across seeds（window 級）')
    print(f'{"設定":<34}{"n":>3}{"mean1":>16}{"lower_mac":>16}'
          f'{"lower_mic":>16}{"upper_mac":>16}')
    for g, ds in groups.items():
        print(f'{g:<34}{len(ds):3d}'
              f'{ms(x["overall"]["mean1"] for x in ds):>16}'
              f'{ms(x["overall"]["lower_mean1"] for x in ds):>16}'
              f'{ms(x["overall"]["lower_top1"] for x in ds):>16}'
              f'{ms(x["overall"]["upper_mean1"] for x in ds):>16}')

    print('\n【B. macro − micro（下半身）】負值越大 = 越靠多數類撐分數')
    for g, ds in groups.items():
        gap = [x['overall']['lower_mean1'] - x['overall']['lower_top1'] for x in ds]
        print(f'  {g:<34}{ms(gap):>16}')

    print('\n【C. 下半身逐類別 recall】mean±std')
    print(f'{"設定":<34}' + ''.join(f'{n:>16}' for n in names))
    for g, ds in groups.items():
        print(f'{g:<34}' + ''.join(
            ms(x['lower']['recall'][i] for x in ds).rjust(16)
            for i in range(nL)))
    print('  支撐樣本數：' + '  '.join(
        f'{names[i]}={D[0]["lower"]["support"][i]}' for i in range(nL)))

    # ── 配對比較 ──────────────────────────────────────────────
    base_key = next((g for g in groups if baseline in g), None)
    if base_key is None:
        print(f'\n[警告] 找不到 baseline「{baseline}」，跳過配對分析。'
              f'可選：{list(groups)}')
        return
    B = {seed_of(x): x for x in groups[base_key]}

    print(f'\n【D. 配對 Δrecall vs {base_key}】')
    print('  同一個 seed、同一批 window 相減，再跨 seed 平均。'
          '這比直接比平均值敏感得多')
    print(f'{"設定":<34}' + ''.join(f'{n:>16}' for n in names) + f'{"Δmacro":>16}')
    for g, ds in groups.items():
        if g == base_key:
            continue
        rows, mac = [], []
        for x in ds:
            s = seed_of(x)
            if s not in B:
                continue
            b = B[s]
            if not aligned(x, b):
                print(f'  [跳過] {x["_file"]} 與 baseline 的驗證集順序不一致')
                continue
            rows.append([x['lower']['recall'][i] - b['lower']['recall'][i]
                         for i in range(nL)])
            mac.append(x['overall']['lower_mean1'] - b['overall']['lower_mean1'])
        if not rows:
            continue
        R = np.array(rows)
        print(f'{g:<34}' + ''.join(
            ms(R[:, i]).rjust(16) for i in range(nL)) + ms(mac).rjust(16))

    print(f'\n【E. McNemar 檢定 vs {base_key}】(下半身，window 級，跨 seed 合併)')
    print('  b = 只有本設定答對   c = 只有 baseline 答對')
    print(f'{"設定":<34}{"b":>8}{"c":>8}{"chi2":>10}{"p":>10}')
    for g, ds in groups.items():
        if g == base_key:
            continue
        b_t = c_t = 0
        for x in ds:
            s = seed_of(x)
            if s not in B or x['_npz'] is None or B[s]['_npz'] is None:
                continue
            if not aligned(x, B[s]):
                continue
            za, zb = x['_npz'], B[s]['_npz']
            ok_a = za['pred_lower'].argmax(1) == za['gt_lower']
            ok_b = zb['pred_lower'].argmax(1) == zb['gt_lower']
            b_t += int((ok_a & ~ok_b).sum())
            c_t += int((~ok_a & ok_b).sum())
        if b_t + c_t == 0:
            continue
        chi = (abs(b_t - c_t) - 1) ** 2 / (b_t + c_t)
        p = np.exp(-chi / 2)          # chi2(1) 上尾的保守近似
        print(f'{g:<34}{b_t:8d}{c_t:8d}{chi:10.3f}{p:10.4f}')
    print('  註：window 之間非獨立（同 clip 多視窗），p 值偏樂觀，只當參考')

    print('\n【F. 消融 Δmacro 對照】')
    keys = sorted({k for d in D for k in d.get('ablation', {}) if k != 'normal'})
    if keys:
        print(f'{"消融":<18}' + ''.join(f'{g.split("/")[0]:>16}' for g in groups))
        for k in keys:
            row = f'{k:<18}'
            for g, ds in groups.items():
                v = [d['overall']['lower_mean1'] - d['ablation'][k]['macro']
                     for d in ds if k in d.get('ablation', {})]
                row += (ms(v).rjust(16) if v else '-'.rjust(16))
            print(row)


def seed_of(d):
    """從 tag 或 ckpt 路徑撈 seed；撈不到就用檔名當 key"""
    import re
    for s in (d.get('tag', ''), d.get('ckpt', ''), d.get('note') or ''):
        m = re.search(r'seed[_-]?(\d+)|s(\d{2})\b', str(s))
        if m:
            return m.group(1) or m.group(2)
    return d['_file']


def aligned(a, b):
    """兩份結果是否落在同一批 window 上（順序也要一樣）"""
    if a['_npz'] is None or b['_npz'] is None:
        return False
    za, zb = a['_npz'], b['_npz']
    return (len(za['gt_lower']) == len(zb['gt_lower'])
            and bool((za['gt_lower'] == zb['gt_lower']).all())
            and bool((za['cid'] == zb['cid']).all()))


# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt')
    ap.add_argument('--cfg', default=CFG)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--note', default='')
    ap.add_argument('--mode', default='', help='留空則從 ckpt 自動推斷')
    ap.add_argument('--inject', default='')
    ap.add_argument('--scope', default='')
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--no-ablation', action='store_true')
    ap.add_argument('--names-lower', default='')
    ap.add_argument('--compare', nargs='*', default=None)
    ap.add_argument('--baseline', default='long_only')
    a = ap.parse_args()

    if a.compare:
        compare(a.compare, a.baseline)
    elif a.ckpt:
        main(a)
    else:
        ap.print_help()
        sys.exit(1)
