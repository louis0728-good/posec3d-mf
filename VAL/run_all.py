# -*- coding: utf-8 -*-
"""
run_all.py — 自動掃描 work_dirs，把所有 checkpoint 餵給 tt2.py

放在 E:\\rcnn\\mmaction2\\ 底下（跟 tt2.py 同一層）：
    python run_all.py --dry             # 先看掃到什麼，不執行
    python run_all.py                   # 只跑逐類別（快）
    python run_all.py --ablation        # 連消融一起跑（慢很多）
    python run_all.py --only 雙門控      # 只跑資料夾名含這個字串的
跑完會自動接 --compare。

為什麼要明確傳 --mode 給 tt2.py：
    dual_window 與 short_only 的參數集合完全一樣（都只有 small_proj、
    沒有 gate），從 state_dict 分不出來。所以 mode 一律從資料夾名決定，
    只讓 tt2.py 自動推斷 motion_inject 與 inject_scope（那兩個可靠）。
"""

import argparse
import glob
import os
import re
import subprocess
import sys

WORK = 'work_dirs'

# 資料夾名（去掉日期前綴與 seed 後綴、去掉運動參數尾巴）→ (neck mode, tag 前綴)
CONFIG2MODE = {
    '純長視窗':      ('long_only',   'longonly'),
    '純短視窗':      ('short_only',  'shortonly'),
    '單門控':        ('per_gate',    'pergate'),
    '雙門控':        ('dual_gate',   'dualgate'),
    '無門控_雙視窗':  ('dual_window', 'dualwindow'),
    '等權相加':      ('dual_window', 'dualwindow'),
    '隨機學習':      ('learnable',   'learnable'),
    '可學習權重':    ('learnable',   'learnable'),
}

MOTION_SUFFIX = {
    '運動參數beta':  'beta',
    '運動參數gamma': 'gamma',
    '運動參數film':  'film',
}

FOLDER_RE = re.compile(r'^(?:\d{8}_)?(?P<name>.+?)_seed[=_-]?(?P<seed>\d+)$')


def parse_folder(folder):
    """→ (mode, motion, tag)；認不得設定名時 mode 回 '?'"""
    m = FOLDER_RE.match(folder)
    if not m:
        return None
    name, seed = m.group('name'), m.group('seed')

    motion = 'none'
    for suf, val in MOTION_SUFFIX.items():
        if name.endswith('_' + suf):
            motion = val
            name = name[:-(len(suf) + 1)]
            break

    if name not in CONFIG2MODE:
        return ('?', name, seed)
    mode, slug = CONFIG2MODE[name]
    return (mode, motion, f'{slug}_m{motion}_seed{seed}')


def find_ckpt(folder, work):
    """挑 best_dual_acc_mean1_epoch_*.pth，排除 upper/lower 那兩個"""
    pat = os.path.join(work, folder, 'best_dual_acc_mean1_epoch_*.pth')
    hits = [p for p in glob.glob(pat)
            if 'upper' not in os.path.basename(p)
            and 'lower' not in os.path.basename(p)]
    if not hits:
        return None

    def ep(p):
        g = re.search(r'epoch_(\d+)', os.path.basename(p))
        return int(g.group(1)) if g else -1

    return max(hits, key=ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ablation', action='store_true')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--only', default='', help='只跑資料夾名含此字串的')
    ap.add_argument('--baseline', default='long_only')
    ap.add_argument('--work', default=WORK)
    a = ap.parse_args()

    folders = sorted(d for d in os.listdir(a.work)
                     if os.path.isdir(os.path.join(a.work, d)))
    if a.only:
        folders = [d for d in folders if a.only in d]

    todo, unknown, nockpt = [], [], []
    for d in folders:
        p = parse_folder(d)
        if p is None:
            unknown.append((d, '名稱不符 <設定>_seed=<n> 格式'))
            continue
        mode, motion, tag = p
        if mode == '?':
            unknown.append((d, f'不認得設定名「{motion}」，請加進 CONFIG2MODE'))
            continue
        ck = find_ckpt(d, a.work)
        if ck is None:
            nockpt.append(d)
            continue
        todo.append((d, mode, motion, tag, ck))

    if unknown:
        print('=== 跳過（名稱無法解析）===')
        for d, why in unknown:
            print(f'  {d}\n      {why}')
        print()
    if nockpt:
        print('=== 跳過（找不到 best_dual_acc_mean1_*.pth）===')
        for d in nockpt:
            print(f'  {d}')
        print()

    print(f'=== 準備跑 {len(todo)} 個 ===')
    print(f'  {"tag":<30}{"mode":<14}ckpt')
    for d, mode, motion, tag, ck in todo:
        print(f'  {tag:<30}{mode:<14}{os.path.basename(ck)}')
    if a.dry:
        return

    ok, fail = [], []
    for i, (d, mode, motion, tag, ck) in enumerate(todo, 1):
        cmd = [sys.executable, 'tt2.py', '--ckpt', ck, '--tag', tag,
               '--mode', mode]
        if not a.ablation:
            cmd.append('--no-ablation')
        print(f'\n{"=" * 70}\n[{i}/{len(todo)}] {tag}   ({d})\n{"=" * 70}')
        r = subprocess.run(cmd)
        (ok if r.returncode == 0 else fail).append(tag)

    print(f'\n\n成功 {len(ok)} / 失敗 {len(fail)}')
    if fail:
        print('失敗的：' + ', '.join(fail))

    files = sorted(glob.glob('diag_*.json'))
    if len(files) >= 2:
        print(f'\n{"=" * 70}\n彙整 {len(files)} 份結果\n{"=" * 70}')
        subprocess.run([sys.executable, 'tt2.py', '--compare', *files,
                        '--baseline', a.baseline])


if __name__ == '__main__':
    main()