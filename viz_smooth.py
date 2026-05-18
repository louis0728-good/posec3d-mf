"""
視覺化：原始 vs 平滑 標籤色帶
用法：python viz_smooth.py
"""

import json, os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.font_manager import FontProperties

CJK_FONT = FontProperties(fname=r'C:\Windows\Fonts\msjh.ttc')
FOOTWORK_MAP = {
    0: '無',
    1: '前進',
    2: '後退',
    3: '長刺',
    4: '飛刺',
    5: '前進長刺',
    6: '後退飛刺',
}

SWORDFIGHT_MAP = {
    0: '無',
    1: '直刺',    
}

TACTIC_MAP = { 
    'None':    '無',
    'Attack':  '攻擊',
    'Remise':  '延續進攻',
    'C-A':     '反擊',
    'Riposte': '還擊',
}

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 劍法暖色系、步法冷色系，一眼分開 ──
EPEE_COLORS = ['#FFCC80', '#E65100']
FOOT_COLORS = ['#B3E5FC', '#1565C0', '#43A047', '#8E24AA',
               '#F9A825', '#D81B60', '#00897B']
NONE_COLOR  = "#0A0000"

TACTIC_COLORS = {
    '攻擊':  '#F44336',
    '延續進攻':  '#FF9800',
    '反擊':     '#FFEB3B',
    '還擊': '#4CAF50',
    '無':    '#0A0000',
}
TACTIC_ORDER = ['無', '攻擊', '延續進攻', '反擊', '還擊']

def build_cmap(colors):
    cmap = ListedColormap([NONE_COLOR] + colors)
    norm = BoundaryNorm(range(-1, len(colors) + 1), cmap.N)
    return cmap, norm


def to_arr(seq):
    return np.array([-1 if v is None else v for v in seq])


def draw_row(ax, seq, cmap, norm, ylabel, total_frames):
    ax.imshow(to_arr(seq).reshape(1, -1), aspect='auto',
              cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_yticks([])
    ax.set_ylabel(ylabel, fontproperties=CJK_FONT, fontsize=12, rotation=0, labelpad=55, va='center') # 左邊那些文字
    ax.set_xlim(-0.5, total_frames - 0.5)
    step = max(1, total_frames // 10)
    ticks = list(range(0, total_frames, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=15) # x 軸數字

def draw_tactic_row(ax, tactics, ylabel, total_frames):
    """把 tactics 列表畫成一行色帶"""
    arr = np.full(total_frames, -1, dtype=float)
    for t in tactics:
        tactic_name = t['tactic']       
        print(f"[debug] 讀到的 tactic_name = {tactic_name!r}")           # 現在是中文
        try:
            idx = TACTIC_ORDER.index(tactic_name)
        except ValueError:
            idx = 0                                   # 找不到就當作「無」
        s = max(0, t['t_start'])
        e = min(total_frames - 1, t['t_end'])
        arr[s:e+1] = idx

    colors = [TACTIC_COLORS[k] for k in TACTIC_ORDER]
    cmap = ListedColormap([NONE_COLOR] + colors)
    norm = BoundaryNorm(range(-1, len(colors) + 1), cmap.N)

    ax.imshow(arr.reshape(1, -1), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_yticks([])
    ax.set_ylabel(ylabel, fontproperties=CJK_FONT, fontsize=12, rotation=0, labelpad=55, va='center')
    ax.set_xlim(-0.5, total_frames - 0.5)
    step = max(1, total_frames // 10)
    ticks = list(range(0, total_frames, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=15)

def viz_one_file(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    persons = data['persons']
    if not persons:
        return
    total_frames = data['total_frames']
    smooth_n = data.get('smooth_n', '?')
    video_name = data.get('video_name', '')

    cmap_e, norm_e = build_cmap(EPEE_COLORS)
    cmap_f, norm_f = build_cmap(FOOT_COLORS)

    # 偵測戰術資料
    tactics_dir = os.path.join(os.path.dirname(json_path), 'tactics')
    tactic_file = os.path.join(tactics_dir, os.path.basename(json_path))
    tactic_data = None
    if os.path.exists(tactic_file):
        with open(tactic_file, 'r', encoding='utf-8') as f:
            tactic_data = json.load(f)

    rows_per_person = 5 if tactic_data is None else 6 # 多兩行：雙方戰術
    n_rows = len(persons) * rows_per_person - 1


    fig, axes = plt.subplots(n_rows, 1,
                         figsize=(16, n_rows * 0.6 + 1.2),
                         constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(f'{video_name}  (N={smooth_n})', fontsize=20) # 總標題

    tactic_map = {}
    if tactic_data:
        for tp in tactic_data.get('persons', []):
            tactic_map[tp['person_id']] = tp['tactics']

    row = 0
    for i, p in enumerate(persons):
        pid = p['person_id']
        axes[row].set_title(f'Person {pid}', fontsize=15, loc='left', pad=2)
        draw_row(axes[row],     p['raw_upper'],      cmap_e, norm_e, '劍法 原始', total_frames)
        draw_row(axes[row + 1], p['smoothed_upper'],  cmap_e, norm_e, '劍法 平滑', total_frames)
        draw_row(axes[row + 2], p['raw_lower'],      cmap_f, norm_f, '步法 原始', total_frames)
        draw_row(axes[row + 3], p['smoothed_lower'],  cmap_f, norm_f, '步法 平滑', total_frames)
        r = row + 4

        if tactic_data and pid in tactic_map:
            draw_tactic_row(axes[r], tactic_map[pid], '戰術', total_frames)
            r += 1

        if i < len(persons) - 1:
            axes[r].axis('off')

        row += rows_per_person

    leg_epee = [mpatches.Patch(color=EPEE_COLORS[i], label=SWORDFIGHT_MAP.get(i, str(i))) for i in range(2)]
    leg_foot = [mpatches.Patch(color=FOOT_COLORS[i], label=FOOTWORK_MAP.get(i, str(i))) for i in range(7)]
    leg_none = [mpatches.Patch(color=NONE_COLOR, label='N/A')]

    leg1 = fig.legend(handles=leg_epee, title='劍法', fontsize=12, title_fontsize=16,
                      loc='lower left', bbox_to_anchor=(0.02, -0.08), ncol=2, frameon=False)
    leg1.get_title().set_fontproperties(CJK_FONT)
 
    leg2 = fig.legend(handles=leg_foot, title='步法', fontsize=12, title_fontsize=16,
                      loc='lower center', bbox_to_anchor=(0.35, -0.08), ncol=4, frameon=False)
    leg2.get_title().set_fontproperties(CJK_FONT)
 
    fig.legend(handles=leg_none, fontsize=12,
               loc='lower right', bbox_to_anchor=(0.95, -0.01), frameon=False)
    fig.add_artist(leg1)
    fig.add_artist(leg2)

    if tactic_data:
        leg_tac = [mpatches.Patch(color=TACTIC_COLORS[k], label=k) 
               for k in TACTIC_ORDER if k != '無']
        leg3 = fig.legend(handles=leg_tac, title='戰術', fontsize=12, title_fontsize=16,
                          loc='lower center', bbox_to_anchor=(0.80, -0.08), ncol=2, frameon=False)
        leg3.get_title().set_fontproperties(CJK_FONT)
        fig.add_artist(leg3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {output_path}')


def main(input_dir=None):
    if input_dir is None:
        input_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'outputs_videos', 'filtering')
    viz_dir = os.path.join(input_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not files:
        print(f'[viz] 找不到 JSON: {input_dir}')
        return
    print(f'[viz] {len(files)} 個檔案')
    for jf in files:
        name = os.path.splitext(os.path.basename(jf))[0]
        viz_one_file(jf, os.path.join(viz_dir, f'{name}.png'))
    print('[viz] 完成')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)
    main(parser.parse_args().input_dir)
