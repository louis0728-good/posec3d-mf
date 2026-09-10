# tt2.py  — 一次跑完所有診斷
import collections, numpy as np, torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint
from mmaction.registry import MODELS, DATASETS

_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})

CFG  = 'configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'
# 每次跑改這兩行，四份權重各跑一次
MODE = 'full'                      # 'full' / 'dual_window_sg' / 'dual_window' / 'sg_only'
CKPT = 'checkpoints/best_full.pth'
NAMES = ['無', '前進', '後退', '長刺', '飛刺', '前進長刺']


def run(model, loader, mode='normal', base_sg=None, base_fs=None):
    """mode: normal / zero_sg / zero_sg_gate / zero_direct / zero_new / zero_fs / zero_both"""
    neck, orig = model.neck, model.neck.forward
    use_fs, use_sg, use_gate = neck.use_fs, neck.use_sg, neck.use_gate
    D = neck.sg_feat_dim // 2 if use_sg else 0
    beta_vecs = []
    pu, pl, gu, gl, cids = [], [], [], [], []
    norms = collections.defaultdict(list)

    def fwd(x, data_samples=None, **kw):
        x2, x3 = x if isinstance(x, (tuple, list)) else (None, x)
        B = x3.shape[0]
        dev = x3.device

        F_L = neck.temporal_pool(neck.spatial_pool(x3)).view(B, -1).float()
        norms['F_L'].append(F_L.norm(dim=1).cpu().numpy())

        F_S = None
        if use_fs:
            m, T2 = neck.crop_margin, x2.shape[2]
            F_S = neck.small_proj(neck.temporal_pool(
                neck.spatial_pool(x2[:, :, m:T2 - m])).view(B, -1)).float()
            norms['F_S'].append(F_S.norm(dim=1).cpu().numpy())
            if mode in ('zero_fs', 'zero_both'):
                F_S = base_fs.to(dev).expand(B, -1).clone()

        sg_n = sg_gate = None
        if use_sg:
            sg = torch.stack([torch.as_tensor(s.sg_features, dtype=torch.float32,
                                              device=dev) for s in data_samples])
            sg_n = neck.sg_norm(sg).float()
            bs = base_sg.to(dev)
            if mode in ('zero_sg', 'zero_both'):
                sg_n = bs.expand(B, -1).clone()
            elif mode == 'zero_new':
                sg_n = sg_n.clone()
                sg_n[:, 34:68]         = bs[34:68]
                sg_n[:, D + 34:D + 68] = bs[D + 34:D + 68]
            sg_gate = bs.expand(B, -1) if mode == 'zero_sg_gate' else sg_n

        if use_gate:
            w = torch.softmax(neck.gating_machine(
                torch.cat([F_L, F_S, sg_gate], 1)).view(B, 2, neck.in_channels), dim=1)
            a, b = w[:, 0], w[:, 1]
            beta_vecs.append(b.detach().cpu().numpy())
            out = a * F_L + b * F_S
            norms['aF_L'].append((a * F_L).norm(dim=1).cpu().numpy())
            norms['bF_S'].append((b * F_S).norm(dim=1).cpu().numpy())
        elif use_fs:
            out = 0.5 * F_L + 0.5 * F_S
        else:
            out = F_L

        if use_sg and mode != 'zero_direct':
            d = neck.sg_gamma * neck.sg_proj(sg_n)
            norms['SGdirect'].append(d.norm(dim=1).cpu().numpy())
            out = out + d
        norms['OUT'].append(out.norm(dim=1).cpu().numpy())
        return out.view(B, neck.out_channels, 1, 1, 1), dict()

    neck.forward = fwd
    try:
        with torch.no_grad():
            for batch in loader:
                for s in model.val_step(batch):
                    pl.append(s.pred_score_lower.cpu().numpy())
                    pu.append(s.pred_score_upper.cpu().numpy())
                    gl.append(int(s.gt_label[1]))
                    gu.append(int(s.gt_label[0]))
                    cids.append(getattr(s, 'frame_dir', '?'))
    finally:
        neck.forward = orig

    bv = np.concatenate(beta_vecs) if beta_vecs else None
    nm = {k: np.concatenate(v) for k, v in norms.items()}
    return bv, np.array(pl), np.array(gl), np.array(pu), np.array(gu), cids, nm


def mca(preds, gts, n=6):
    p = preds.argmax(1)
    per = [float((p[gts == c] == c).mean()) if (gts == c).any() else 0. for c in range(n)]
    return float(np.mean(per)), per


def templates(bv, gts):
    M = np.stack([bv[gts == c].mean(0) for c in range(6)])
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    Mc = M - M.mean(0, keepdims=True)                    # 置中相關：去掉共同成分
    Mc /= np.linalg.norm(Mc, axis=1, keepdims=True)
    return Mn @ Mn.T, Mc @ Mc.T


if __name__ == '__main__':
    init_default_scope('mmaction')
    cfg = Config.fromfile(CFG)
    cfg.model.neck.mode = MODE
    cfg.model.neck.debug = False
    cfg.val_dataloader.num_workers = 0
    cfg.val_dataloader.persistent_workers = False
    loader = Runner.build_dataloader(cfg.val_dataloader)
    ds = DATASETS.build(cfg.val_dataloader.dataset)

    model = MODELS.build(cfg.model).cuda().eval()
    load_checkpoint(model, CKPT, map_location='cpu')
    g = model.neck.sg_gamma.detach().cpu()
    print(f'sg_gamma shape = {tuple(g.shape)}')
    print(f'  正值: {(g > 0).sum().item()}  負值: {(g < 0).sum().item()}  接近0: {(g.abs() < 0.01).sum().item()}')
    print(f'  mean={g.mean().item():.4f}  std={g.std().item():.4f}  範圍=[{g.min().item():.4f}, {g.max().item():.4f}]')
    model.neck.debug = False

    bv,  P,  G,  C  = run(model, loader, 'normal')
    bv0, P0, _,  _  = run(model, loader, 'zero_sg')
    _,   P1, _,  _  = run(model, loader, 'zero_fs')
    _,   P2, _, _  = run(model, loader, 'zero_new') 
    _,   P3, _, _  = run(model, loader, 'zero_direct')
    m,  per  = mca(P,  G)
    m0, per0 = mca(P0, G)
    m1, per1 = mca(P1, G)
    m2, per2 = mca(P2, G)   # zero_new
    m3, per3 = mca(P3, G)   # zero_direct

    # ---------- 1. 混淆矩陣 + 校準 ----------
    pc = P.argmax(1)
    cm = np.zeros((6, 6), int)
    for t, p in zip(G, pc): cm[t, p] += 1
    print('\n===== 1. 混淆矩陣（列=真實 行=預測）=====')
    print(f'{"":12}' + ''.join(f'{n:>8}' for n in NAMES) + f'{"recall":>9}{"預測/真實":>11}')
    for i in range(6):
        print(f'{NAMES[i]:<12}' + ''.join(f'{cm[i,j]:8d}' for j in range(6))
              + f'{per[i]:9.3f}{cm[:,i].sum()/max(cm[i].sum(),1):11.3f}')
    print(f'lower_mean1 = {m:.4f}')

    # ---------- 2. 消融（含新增兩組）----------
    print('\n===== 2. 消融：原始 / zeroSG / zeroF_S / zero_new / zero_direct =====')
    print(f'{"類別":<12}{"原始":>8}{"zSG":>8}{"zFS":>8}{"zNew":>8}{"zDir":>8}'
        f'{"ΔSG":>8}{"ΔFS":>8}{"ΔNew":>8}{"ΔDir":>8}')
    for i in range(6):
        print(f'{NAMES[i]:<12}'
            f'{per[i]:8.3f}{per0[i]:8.3f}{per1[i]:8.3f}{per2[i]:8.3f}{per3[i]:8.3f}'
            f'{per[i]-per0[i]:+8.3f}{per[i]-per1[i]:+8.3f}'
            f'{per[i]-per2[i]:+8.3f}{per[i]-per3[i]:+8.3f}')
    print(f'{"mean1":<12}'
        f'{m:8.4f}{m0:8.4f}{m1:8.4f}{m2:8.4f}{m3:8.4f}'
        f'{m-m0:+8.4f}{m-m1:+8.4f}{m-m2:+8.4f}{m-m3:+8.4f}')

    # ---------- 3. β 樣板（含置中相關）----------
    s_raw, s_ctr   = templates(bv,  G)
    s_raw0, s_ctr0 = templates(bv0, G)
    off = ~np.eye(6, dtype=bool)
    print('\n===== 3. β 樣板相似度 =====')
    print('原始（置中相關）:\n', np.round(s_ctr, 3))
    print('zeroSG（置中相關）:\n', np.round(s_ctr0, 3))
    print(f'\n群間平均  raw: {s_raw[off].mean():.4f} → {s_raw0[off].mean():.4f}'
          f'   置中: {s_ctr[off].mean():.4f} → {s_ctr0[off].mean():.4f}')
    print(f'逐通道 β std  原始 {bv.std(0).mean():.4f} / zeroSG {bv0.std(0).mean():.4f}')
    print(f'β 全體平均 = {bv.mean():.4f}')
    print(f'各樣本 β 均值的標準差 = {bv.mean(1).std():.4f}')

    # ---------- 4. clip 級錯誤 + 按類別/視窗數分解 ----------
    byclip = collections.defaultdict(list)
    for cid, p, g in zip(C, pc, G): byclip[cid].append((p == g, g))
    stat = collections.defaultdict(lambda: [0, 0])      # [全錯, 總數]
    wstat = collections.defaultdict(lambda: [0, 0])     # 依視窗數
    full, part, ok = 0, 0, 0
    for cid, v in byclip.items():
        nw, nw_ok, g = len(v), sum(x[0] for x in v), v[0][1]
        stat[g][1] += 1; wstat[nw][1] += 1
        if nw_ok == 0:   full += 1; stat[g][0] += 1; wstat[nw][0] += 1
        elif nw_ok == nw: ok += 1
        else:            part += 1
    n = len(byclip)
    print(f'\n===== 4. Clip 級 =====\n全對 {ok} ({ok/n:.1%}) 部分錯 {part} ({part/n:.1%}) 全錯 {full} ({full/n:.1%})')
    print(f'\n{"類別":<12}{"全錯":>7}{"總clip":>8}{"全錯率":>9}')
    for i in range(6):
        print(f'{NAMES[i]:<12}{stat[i][0]:7d}{stat[i][1]:8d}{stat[i][0]/max(stat[i][1],1):9.1%}')
    print(f'\n{"視窗數":<10}{"全錯":>7}{"clip數":>8}{"全錯率":>9}')
    for k in sorted(wstat):
        print(f'{k:<10}{wstat[k][0]:7d}{wstat[k][1]:8d}{wstat[k][0]/max(wstat[k][1],1):9.1%}')

    # ---------- 5. 「無」的髖部位移（決定要不要拆）----------
    SH, HP = [5, 6], [11, 12]
    feat = []
    for i in range(len(ds)):
        kp = ds.get_data_info(i)['keypoint'].astype(np.float32)
        sh, hp = kp[:, :, SH].mean(2)[0], kp[:, :, HP].mean(2)[0]
        torso = max(float(np.median(np.linalg.norm(sh - hp, axis=-1))), 1e-3)
        net  = float(np.linalg.norm(hp[-1] - hp[0])) / torso
        path = float(np.sum(np.linalg.norm(np.diff(hp, axis=0), axis=-1))) / torso
        feat.append((net, path, net / max(path, 1e-6)))
    feat = np.array(feat)
    assert len(feat) == len(G), '順序對不上，檢查 shuffle 是否為 False'

    print('\n===== 5. 髖部位移（torso 為單位）=====')
    print(f'{"類別":<12}{"淨位移":>9}{"路徑長":>9}{"直線度":>9}')
    for i in range(6):
        f = feat[G == i]
        print(f'{NAMES[i]:<12}{f[:,0].mean():9.3f}{f[:,1].mean():9.3f}{f[:,2].mean():9.3f}')

    w = (G == 0); corr = w & (pc == 0)
    print(f'\n「無」正確 vs 錯誤（n={w.sum()}）:')
    for nm, k in (('淨位移', 0), ('路徑長', 1), ('直線度', 2)):
        a, b = feat[corr][:, k], feat[w & ~corr][:, k]
        print(f'  {nm}: 正確 {a.mean():.3f}±{a.std():.3f}   錯誤 {b.mean():.3f}±{b.std():.3f}')
    q = np.percentile(feat[w][:, 1], [10, 25, 50, 75, 90])
    print(f'  「無」路徑長分位數 (10/25/50/75/90): {np.round(q,3)}')

    # ---------- 6. 信心分佈 ----------
    conf = P.max(1)
    hit = (pc == G)
    print(f'\n===== 6. 信心 =====\n正確 {conf[hit].mean():.3f}±{conf[hit].std():.3f}'
          f'   錯誤 {conf[~hit].mean():.3f}±{conf[~hit].std():.3f}')
    for t in (0.5, 0.7, 0.9):
        k = conf >= t
        print(f'  閾值 {t}: 覆蓋 {k.mean():.1%}  該子集正確率 {hit[k].mean():.3f}')