# -*- coding: utf-8 -*-
r"""
tactic_pipeline.py  —  放在 E:\rcnn\mmaction2\（跟 movement_inference.py 同層）

把原本三支合成一支：
    smoothing.py        濾波(多數決) + 聚合成區塊
    tactic_labeling.py  條件式戰術判定
    final_package.py    打包成前端要的格式

輸入：<result_base>\<比賽>\<回合>\<回合>_action.json   ← movement_inference.py 的輸出
輸出：<result_base>\<比賽>\<回合>\<回合>_final.json    ← app.py /api/tactics 讀這個

用法:
    python tactic_pipeline.py                       # 全部比賽
    python tactic_pipeline.py --match 0             # 只跑這一場
    python tactic_pipeline.py --match 0 --overwrite # 已有 _final.json 也重算
    python tactic_pipeline.py --keep-intermediate   # 順便留下濾波/戰術中間檔,除錯用

與原本三支的差異(只有這三點,演算法一行都沒動):
  1. 走巢狀結構,不再平掃單一資料夾 —— 不同場的 test001 不會再互相覆蓋。
  2. 三段在記憶體裡串,不落地中間檔(要的話加 --keep-intermediate)。
     原本 smoothing 呼叫 run() / tactic_labeling 呼叫 final_run() 都沒把
     input_dir 傳下去,只要目錄不是預設值,第二棒就會斷掉 —— 這裡不會有這問題。
  3. 戰術字串直接輸出顯示名(進攻 / 反攻 / 無動靜),見下方 TACTIC_MAP 的註解。
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

# =====================================================================
#  參數(沿用原本三支的預設值)
# =====================================================================
RESULT_BASE = r"D:\project\action_output_videos"

SMOOTH_N = 3            # 多數決重複次數           (smoothing.py)
MIN_DUR = 0              # 短於這麼多幀的區塊視為雜訊,吸收進鄰居。0 = 關閉
WINDOW_SIZE = 21         # 對齊 PoseC3D 的 clip_len (smoothing.py)

OMEGA_ATK = {1}          # label_upper 中屬於攻擊的值 (tactic_labeling.py)
REMISE_THETA = 15        # Remise 判斷門檻
RIPOSTE_GAP_TOLERATE = 5 # Riposte 允許時間差(幀)
CA_TOLERANCE = 5         # C-A 判斷:我比對手晚出手幾幀才算被動

#  原本是 {'None':'無', 'C-A':'反擊', 'Attack':'攻擊'}。
#  block_b.js 有 RAW2TAC、block_c.js 的 TAC 有 alias,兩種寫法都吃得下;
#  但 block_e.js 的 TACTIC_KEYS 只認顯示名,戰術過濾會整組比不到。
#  這裡直接輸出顯示名,三個 block 都不用改。
TACTIC_MAP = {
    'None':    '無動靜',
    'C-A':     '反攻',
    'Remise':  '延續進攻',
    'Riposte': '還擊',
    'Attack':  '進攻',
}

#  label 數字 → 名稱。這份只是對照參考,JSON 存的是數字,
#  真正畫在畫面上的名稱在 block_b.js / block_c.js 的 EPEE / FOOT。
UPPER_LABELS = {0: '無', 1: '直刺'}
LOWER_LABELS = {0: '無', 1: '前進', 2: '後退', 3: '長刺', 4: '飛刺', 5: '前進長刺'}


# =====================================================================
#  第一段:濾波 + 聚合   (原 smoothing.py)
# =====================================================================
def flatten(segments, total_frames, window_size=WINDOW_SIZE, assign='span'):
    """
    滑動視窗結果攤平成逐幀序列。

    assign='span'（預設，與 movement_inference.py 疊字影片同一套規則）
        把每個視窗覆蓋的 21 幀全部標上,重疊處取 (score_upper+score_lower)/2
        最高的那個視窗。第 0 幀開始就有值,而且圖上看到的跟影片上疊的字逐幀一致。

    assign='mid'（舊行為）
        每個視窗只認中間那一幀(start_frame + 10)。前後各 10 幀會是 None,
        所以圖的左邊會缺一小段,也跟影片對不起來。
    """
    persons = sorted(set(seg['person_id'] for seg in segments))
    result = {}

    for pid in persons:
        upper = [None] * total_frames
        lower = [None] * total_frames
        pid_segs = [s for s in segments if s['person_id'] == pid]

        if assign == 'mid':
            mid = window_size // 2
            for seg in pid_segs:
                acting = seg['start_frame'] + mid
                if 0 <= acting < total_frames:
                    upper[acting] = seg['label_upper']
                    lower[acting] = seg['label_lower']
        else:
            best = [None] * total_frames        # 每幀目前分數最高的那個視窗
            for seg in pid_segs:
                sc = (seg.get('score_upper', 0) + seg.get('score_lower', 0)) / 2.0
                start = max(0, seg['start_frame'])
                end = min(seg.get('end_frame', seg['start_frame'] + window_size), total_frames)
                for i in range(start, end):
                    if best[i] is None or sc > best[i][0]:
                        best[i] = (sc, seg)
            for i, b in enumerate(best):
                if b is not None:
                    upper[i] = b[1]['label_upper']
                    lower[i] = b[1]['label_lower']

        result[pid] = {'upper': upper, 'lower': lower}

    return result


def mj_vote(seq, n_times):
    """
    窗口大小 3（t-1, t, t+1）的多數決投票,重複 N 次。
    - None 的幀不參與投票,也不被修改。
    - 平票時保持原值 t 不變。
    """
    arr = list(seq)

    for _ in range(n_times):
        new_arr = list(arr)
        for t in range(len(arr)):
            if arr[t] is None:
                continue

            smooth_window = []
            for dt in range(-2, 3):   # 平滑視窗
                idx = t + dt
                if 0 <= idx < len(arr) and arr[idx] is not None:
                    smooth_window.append(arr[idx])

            if not smooth_window:
                continue

            counter = Counter(smooth_window)
            most_common = counter.most_common()

            # 有唯一最大值(非平票)才改
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                new_arr[t] = most_common[0][0]

        arr = new_arr

    return arr


def we_are_freinds(seq):
    """
    將連續且標籤一致的幀聚合為區塊。跳過 None 幀。
    回傳 list of {label, t_start, t_end}
    """
    blocks = []
    current_label = None
    t_start = None

    for t, label in enumerate(seq):
        if label is None:
            if current_label is not None:
                blocks.append({'label': current_label, 't_start': t_start, 't_end': t - 1})
                current_label = None
                t_start = None
            continue

        if label == current_label:
            continue
        else:
            if current_label is not None:
                blocks.append({'label': current_label, 't_start': t_start, 't_end': t - 1})
            current_label = label
            t_start = t

    if current_label is not None:
        blocks.append({'label': current_label, 't_start': t_start, 't_end': len(seq) - 1})

    return blocks


def drop_short(blocks, min_dur):
    """
    把短於 min_dur 幀的區塊吸收進「時間上相鄰」的較長鄰居。

    為什麼要這一步:PoseC3D 的滑動視窗是 21 幀,任何比它短很多的區塊都不是
    獨立證據,而是視窗邊界上 argmax 跳動的產物。畫在三軌圖上會變成一整片
    看不懂的細條紋。

    - 每次處理當下最短的那一段,由短到長,結果與處理順序無關。
    - 只吸收真正相鄰的(t_end + 1 == t_start),不會跨過 None 造成的空隙。
    - 吸收後相鄰同標籤自動合併。時間軸涵蓋範圍不變,不會憑空少掉一段。
    """
    if min_dur <= 1 or not blocks:
        return [dict(b) for b in blocks]

    b = [dict(x) for x in blocks]

    def dur(x):
        return x['t_end'] - x['t_start'] + 1 if x else -1

    def coalesce(lst):
        out = [lst[0]]
        for x in lst[1:]:
            if x['label'] == out[-1]['label'] and x['t_start'] <= out[-1]['t_end'] + 1:
                out[-1]['t_end'] = max(out[-1]['t_end'], x['t_end'])
            else:
                out.append(x)
        return out

    while len(b) > 1:
        idx = None
        for i, x in enumerate(b):
            if dur(x) >= min_dur:
                continue
            prev_ok = i > 0 and b[i - 1]['t_end'] + 1 == x['t_start']
            next_ok = i + 1 < len(b) and x['t_end'] + 1 == b[i + 1]['t_start']
            if (prev_ok or next_ok) and (idx is None or dur(x) < dur(b[idx])):
                idx = i
        if idx is None:
            break

        x = b[idx]
        prev = b[idx - 1] if idx > 0 and b[idx - 1]['t_end'] + 1 == x['t_start'] else None
        nxt = b[idx + 1] if idx + 1 < len(b) and x['t_end'] + 1 == b[idx + 1]['t_start'] else None
        target = prev if dur(prev) >= dur(nxt) else nxt

        target['t_start'] = min(target['t_start'], x['t_start'])
        target['t_end'] = max(target['t_end'], x['t_end'])
        b.pop(idx)
        b = coalesce(b)

    return b


def smooth_and_block(segments, total_frames, smooth_n=SMOOTH_N, window_size=WINDOW_SIZE,
                     min_dur=MIN_DUR, assign='span'):
    """一部影片的第一段完整流程。回傳 list of person dict(格式同原本的 filtering json)。"""
    per_person = flatten(segments, total_frames, window_size, assign)
    out = []

    for pid in sorted(per_person.keys()):
        raw_upper = per_person[pid]['upper']
        raw_lower = per_person[pid]['lower']

        smoothed_upper = mj_vote(raw_upper, smooth_n)
        smoothed_lower = mj_vote(raw_lower, smooth_n)

        out.append({
            'person_id': pid,
            'blocks_upper': drop_short(we_are_freinds(smoothed_upper), min_dur),
            'blocks_lower': drop_short(we_are_freinds(smoothed_lower), min_dur),
            'raw_upper': raw_upper,
            'raw_lower': raw_lower,
            'smoothed_upper': smoothed_upper,
            'smoothed_lower': smoothed_lower,
        })

    return out


# =====================================================================
#  第二段:戰術判定   (原 tactic_labeling.py)
# =====================================================================
def label_at(blocks, t):
    for b in blocks:
        if b['t_start'] <= t <= b['t_end']:
            return b['label']
    return None


def orig_block_at(blocks, t):
    for b in blocks:
        if b['t_start'] <= t <= b['t_end']:
            return b
    return None


def has_lower_action(blocks_lower, must_start, must_end):
    """檢查時序區間內是否有下半身動作 (label != 0)"""
    for b in blocks_lower:
        overlap_start = max(b['t_start'], must_start)
        overlap_end = min(b['t_end'], must_end)
        if overlap_start <= overlap_end:
            if b['label'] != 0:
                return True
    return False


def prev_opp_attack(opp_atk_blocks, t_start, riposte_gap_tolerate):
    best = None
    for ob in opp_atk_blocks:
        if ob['t_end'] < t_start:
            if best is None or ob['t_end'] > best['t_end']:
                best = ob
    if best is not None and (t_start - best['t_end']) <= riposte_gap_tolerate:
        return best
    return None


def assign_tactics(blocks_upper_h, blocks_upper_opp, blocks_lower_h,
                   remise_theta=REMISE_THETA,
                   riposte_gap_tolerate=RIPOSTE_GAP_TOLERATE,
                   ca_tolerance=CA_TOLERANCE):
    """依計畫書的條件式邏輯判定。h = 本人, opp = 對手。"""
    atk_h = [b for b in blocks_upper_h if b['label'] in OMEGA_ATK]      # 目前只有直刺 label:1
    atk_opp = [b for b in blocks_upper_opp if b['label'] in OMEGA_ATK]

    boundaries = set()
    for b in blocks_upper_h + blocks_upper_opp:
        boundaries.add(b['t_start'])
        boundaries.add(b['t_end'] + 1)          # 右端用開區間收集
    boundaries = sorted(boundaries)

    if len(boundaries) < 2:
        return []

    tactics = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1] - 1

        h_label = label_at(blocks_upper_h, s)
        opp_label = label_at(blocks_upper_opp, s)

        # 非攻擊 → None
        if h_label is None or h_label not in OMEGA_ATK:
            tactics.append({'tactic': TACTIC_MAP['None'], 't_start': s, 't_end': e})
            continue

        orig = orig_block_at(blocks_upper_h, s)     # 為了 Remise(要知道原始長度)
        orig_dur = (orig['t_end'] - orig['t_start']) if orig else 0

        # C-A：此片段中對手也在攻擊,而且我比較晚出手(被動)
        if opp_label is not None and opp_label in OMEGA_ATK:
            opp_orig = orig_block_at(blocks_upper_opp, s)
            if orig and opp_orig:
                if (orig['t_start'] - opp_orig['t_start']) >= ca_tolerance:
                    tactics.append({
                        'tactic': TACTIC_MAP['C-A'],
                        't_start': s, 't_end': e,
                        'overlap': {'t_start': s, 't_end': e},
                    })
                    continue

        # Remise：原始區塊總長 >= THETA,且前 1/3 區段有下半身動作
        if orig_dur >= remise_theta:
            t_third = orig['t_start'] + (orig_dur / 3.0)
            if has_lower_action(blocks_lower_h, orig['t_start'], t_third):
                tactics.append({'tactic': TACTIC_MAP['Remise'], 't_start': s, 't_end': e})
                continue

        # Riposte：對手剛攻完、間隔小
        orig_start = orig['t_start'] if orig else s
        prev = prev_opp_attack(atk_opp, orig_start, riposte_gap_tolerate)
        if prev is not None:
            tactics.append({'tactic': TACTIC_MAP['Riposte'], 't_start': s, 't_end': e})
            continue

        # Attack：預設
        tactics.append({'tactic': TACTIC_MAP['Attack'], 't_start': s, 't_end': e})

    if not tactics:
        return []

    # 相鄰同戰術合併
    merged = [tactics[0].copy()]
    for t in tactics[1:]:
        prev = merged[-1]
        if t['tactic'] == prev['tactic']:
            prev['t_end'] = t['t_end']
            if 'overlap' in t and 'overlap' in prev:
                prev['overlap']['t_end'] = t['overlap']['t_end']
        else:
            merged.append(t.copy())
    return merged


def label_tactics(persons_blocks, remise_theta=REMISE_THETA,
                  riposte_gap_tolerate=RIPOSTE_GAP_TOLERATE,
                  ca_tolerance=CA_TOLERANCE):
    """
    一部影片的第二段。需要兩個人才能分析互動戰術,不足回傳 None
    （與原本 tactic_labeling.process_file 的行為一致）。
    """
    if len(persons_blocks) < 2:
        return None

    upper_block = {p['person_id']: p['blocks_upper'] for p in persons_blocks}
    lower_block = {p['person_id']: p['blocks_lower'] for p in persons_blocks}
    pids = sorted(upper_block.keys())
    pid_a, pid_b = pids[0], pids[1]

    return {
        pid_a: assign_tactics(upper_block[pid_a], upper_block[pid_b], lower_block[pid_a],
                              remise_theta, riposte_gap_tolerate, ca_tolerance),
        pid_b: assign_tactics(upper_block[pid_b], upper_block[pid_a], lower_block[pid_b],
                              remise_theta, riposte_gap_tolerate, ca_tolerance),
    }


# =====================================================================
#  第三段:打包成前端格式   (原 final_package.py)
# =====================================================================
def check_points(blocks, static_start, static_end):
    """把區塊裁切到 [static_start, static_end] 之內。"""
    result = []
    for b in blocks:
        if b['t_end'] < static_start or b['t_start'] > static_end:
            continue
        result.append({
            'label': b['label'],
            't_start': max(b['t_start'], static_start),
            't_end':   min(b['t_end'],   static_end),
        })
    return result


def pack_person(tactics, blocks_upper, blocks_lower):
    """每個戰術段配上該段內的劍法/步法區塊。（原 final_package.abc）"""
    tactic_results = []
    for t in tactics:
        tau_s = t['t_start']
        tau_e = t['t_end']

        entry = {
            'tactic': t['tactic'],
            'tau_start': tau_s,
            'tau_end': tau_e,
            'Y_upper': check_points(blocks_upper, tau_s, tau_e),
            'Y_lower': check_points(blocks_lower, tau_s, tau_e),
        }
        if 'overlap' in t:
            entry['overlap'] = t['overlap']

        tactic_results.append(entry)

    return tactic_results


# =====================================================================
#  三段串起來
# =====================================================================
def build_final(action_data, smooth_n=SMOOTH_N, window_size=WINDOW_SIZE,
                remise_theta=REMISE_THETA, riposte_gap_tolerate=RIPOSTE_GAP_TOLERATE,
                ca_tolerance=CA_TOLERANCE, min_dur=MIN_DUR, assign='span'):
    """
    吃 movement_inference.py 的 _action.json（dict），
    吐 app.py /api/tactics 要的格式（dict）。
    人數不足回傳 (None, blocks, None)。
    """
    segments = action_data.get('segments', [])
    total_frames = action_data.get('total_frames', 0)

    if not segments or not total_frames:
        return None, None, None

    # ① 濾波 + 聚合
    persons_blocks = smooth_and_block(segments, total_frames, smooth_n, window_size,
                                      min_dur, assign)

    # ② 戰術判定
    tactic_map = label_tactics(persons_blocks, remise_theta, riposte_gap_tolerate, ca_tolerance)
    if tactic_map is None:
        return None, persons_blocks, None

    # ③ 打包
    block_map = {p['person_id']: p for p in persons_blocks}
    persons_result = []
    for pid in sorted(block_map.keys()):
        if pid not in tactic_map:
            continue
        persons_result.append({
            'person_id': pid,
            'tactic_results': pack_person(tactic_map[pid],
                                          block_map[pid]['blocks_upper'],
                                          block_map[pid]['blocks_lower']),
        })

    final = {
        'video_name':   action_data.get('video_name', ''),
        'fps':          action_data.get('fps', 30.0),
        'total_frames': total_frames,
        'persons':      persons_result,
    }
    return final, persons_blocks, tactic_map


# =====================================================================
#  走訪 <result_base>\<比賽>\<回合>\
# =====================================================================
def iter_rounds(result_base, match=None):
    """yield (比賽, 回合, _action.json 的路徑)"""
    if not os.path.isdir(result_base):
        return

    matches = sorted(d for d in os.listdir(result_base)
                     if os.path.isdir(os.path.join(result_base, d)))
    if match:
        matches = [d for d in matches if d == match]

    for m in matches:
        mdir = os.path.join(result_base, m)
        for rdir in sorted(glob.glob(os.path.join(mdir, '*'))):
            if not os.path.isdir(rdir):
                continue
            rnd = os.path.basename(rdir)
            action_json = os.path.join(rdir, f'{rnd}_action.json')
            if os.path.exists(action_json):
                yield m, rnd, action_json


def process_round(action_json, out_dir, rnd, args):
    """回傳 'ok' / 'skip:原因' / 'fail:原因'"""
    final_path = os.path.join(out_dir, f'{rnd}_final.json')

    if os.path.exists(final_path) and not args.overwrite:
        return 'skip:已存在'

    with open(action_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    final, blocks, tactic_map = build_final(
        data, args.smooth_n, args.window_size,
        args.remise_theta, args.riposte_gap, args.ca_tolerance, args.min_dur, args.assign)

    if final is None:
        if blocks is None:
            return 'skip:沒有 segments 或 total_frames 是 0'
        return f'skip:只有 {len(blocks)} 個人,互動戰術需要兩人'

    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    if args.keep_intermediate:
        with open(os.path.join(out_dir, f'{rnd}_filtering.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'video_name': data.get('video_name', ''),
                'fps': data.get('fps', 30.0),
                'total_frames': data.get('total_frames', 0),
                'window_size': args.window_size,
                'smooth_n': args.smooth_n,
                'assign': args.assign,
                'min_dur': args.min_dur,
                'persons': blocks,
            }, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, f'{rnd}_tactics.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'video_name': data.get('video_name', ''),
                'remise_theta': args.remise_theta,
                'riposte_gap_tolerate': args.riposte_gap,
                'ca_tolerance': args.ca_tolerance,
                'persons': [{'person_id': pid, 'tactics': tactic_map[pid]}
                            for pid in sorted(tactic_map)],
            }, f, indent=2, ensure_ascii=False)

    n = sum(len(p['tactic_results']) for p in final['persons'])
    return f'ok:{len(final["persons"])} 人 / {n} 個戰術段'


def parse_args():
    ap = argparse.ArgumentParser(description='濾波 + 戰術判定 + 打包（原三支合一）')
    ap.add_argument('--result_base', default=RESULT_BASE,
                    help='動作辨識輸出根目錄,底下是 <比賽>\\<回合>\\')
    ap.add_argument('--match', default=None, help='只處理這一場（子資料夾名）,不給就全部')
    ap.add_argument('--overwrite', action='store_true', help='已有 _final.json 也重算')
    ap.add_argument('--keep-intermediate', action='store_true',
                    dest='keep_intermediate',
                    help='順便寫出 _filtering.json / _tactics.json,除錯用')
    ap.add_argument('--min-dur', type=int, default=MIN_DUR, dest='min_dur',
                    help='短於這麼多幀的區塊吸收進鄰居,清掉三軌圖上的細條紋。'
                         '0=關閉(維持原行為),建議 7')
    ap.add_argument('--assign', choices=['span', 'mid'], default='span',
                    help="span=視窗覆蓋的每一幀都標(同疊字影片,預設); mid=只認視窗中點(舊行為)")
    ap.add_argument('--smooth-n', type=int, default=SMOOTH_N, dest='smooth_n')
    ap.add_argument('--window-size', type=int, default=WINDOW_SIZE, dest='window_size')
    ap.add_argument('--remise-theta', type=int, default=REMISE_THETA, dest='remise_theta')
    ap.add_argument('--riposte-gap', type=int, default=RIPOSTE_GAP_TOLERATE, dest='riposte_gap')
    ap.add_argument('--ca-tolerance', type=int, default=CA_TOLERANCE, dest='ca_tolerance')
    return ap.parse_args()


def main():
    args = parse_args()

    rounds = list(iter_rounds(args.result_base, args.match))
    if not rounds:
        where = f'{args.result_base}\\{args.match}' if args.match else args.result_base
        print(f'[打包] 在 {where} 底下找不到任何 _action.json')
        print(f'[打包] 確認 movement_inference.py 跑過了,而且輸出是 <比賽>\\<回合>\\<回合>_action.json')
        sys.exit(1)

    print(f'[打包] 找到 {len(rounds)} 個回合待處理\n')

    done = skipped = 0
    failed = []
    cur_match = None

    for m, rnd, action_json in rounds:
        if m != cur_match:
            cur_match = m
            print(f'=== 比賽 {m} ===')

        out_dir = os.path.dirname(action_json)
        try:
            status = process_round(action_json, out_dir, rnd, args)
        except Exception as e:
            print(f'  [失敗] {rnd}: {type(e).__name__}: {e}')
            failed.append(f'{m}/{rnd}')
            continue

        tag, _, detail = status.partition(':')
        if tag == 'ok':
            done += 1
            print(f'  [完成] {rnd}  ({detail})')
        else:
            skipped += 1
            print(f'  [跳過] {rnd}  ({detail})')

    print(f'\n[打包] 完成 {done} 個,跳過 {skipped} 個,失敗 {len(failed)} 個')
    if failed:
        for f in failed:
            print(f'  - {f}')
        sys.exit(1)

    if done == 0 and skipped > 0:
        print('[打包] 全部都跳過了。要重算請加 --overwrite')


if __name__ == '__main__':
    main()
