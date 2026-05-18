import json, os, glob, argparse
OMEGA_ATK = {1} # label_upper 中屬於攻擊的值
TACTIC_MAP = {
    'None':    '無',
    'C-A':     '反擊',
    'Remise':  '延續進攻',
    'Riposte': '還擊',
    'Attack':  '攻擊',
}
REMISE_THETA = 15  # Remise 判斷門檻
RITPOSE_GAP_TOLERATE = 5  # Riposte 允許時間差（幀）
CA_TOLERANCE = 5 
from final_package import final_run

def label_at(blocks, t):
    for b in blocks:
        if b['t_start'] <= t <= b['t_end']: # 剛好處於 t_start, t_end 的 那個 block
            return b['label']
    return None

def orig_block_at(blocks, t):
    for b in blocks:
        if b['t_start'] <= t <= b['t_end']:
            return b # 連帶著 t_start, t_end
    return None

def has_lower_action(blocks_lower, must_start, must_end):
    """檢查在時序區間內是否有下半身動作 (label != 0 = None)"""
    for b in blocks_lower:
        # 判斷兩個時間區段是否有交集
        overlap_start = max(b['t_start'], must_start)
        overlap_end = min(b['t_end'], must_end)
        if overlap_start <= overlap_end:
            if b['label'] != 0:  # 如果不是 None
                return True
    return False

def prev_opp_attack(opp_atk_blocks, t_start, ritpose_gap_tolerate):
    best = None
    for ob in opp_atk_blocks:
        if ob['t_end'] < t_start:
            if best is None or ob['t_end'] > best['t_end']:
                best = ob 
    if best is not None and (t_start - best['t_end']) <= ritpose_gap_tolerate:
        return best
    return None

def assign_tactics(blocks_upper_h, blocks_upper_opp, blocks_lower_h, remisze_theta=REMISE_THETA, ritpose_gap_tolerate=RITPOSE_GAP_TOLERATE):
    # 如我計畫書的邏輯判斷
    atk_h   = [b for b in blocks_upper_h   if b['label'] in OMEGA_ATK] # 目前只有直刺 label: 1
    atk_opp = [b for b in blocks_upper_opp if b['label'] in OMEGA_ATK]

    boundaries = set()
    for b in blocks_upper_h + blocks_upper_opp:
        boundaries.add(b['t_start'])
        boundaries.add(b['t_end'] + 1)      # 右端用開區間收集
    boundaries = sorted(boundaries) # 把所有開始和結束時間都 hash 在一起，反正鄰與鄰之間會切一刀

    if len(boundaries) < 2:
        return []

    tactics = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1] - 1 

        h_label   = label_at(blocks_upper_h, s) # 每個 t_start, t_end (其實就是 boundaries) 的 label 
        # 這裡會代表剛好在那個 s, e 區間的  blocks: label
        opp_label = label_at(blocks_upper_opp, s)

        # 非攻擊  None
        if h_label is None or h_label not in OMEGA_ATK:
            tactics.append({'tactic': TACTIC_MAP['None'], 't_start': s, 't_end': e})
            continue

        orig = orig_block_at(blocks_upper_h, s) # 為了 REMISE (因為要知道原始長度)
        orig_dur = (orig['t_end'] - orig['t_start']) if orig else 0

        # C-A：此片段中對手也在攻擊
        if opp_label is not None and opp_label in OMEGA_ATK:
            opp_orig = orig_block_at(blocks_upper_opp, s)
            if orig and opp_orig:
                if (orig['t_start'] - opp_orig['t_start']) >= CA_TOLERANCE:
                    # 我較晚出手 (被動)，判定為 C-A
                    tactics.append({
                        'tactic': TACTIC_MAP['C-A'],    # ← 改這裡
                        't_start': s, 't_end': e,
                        'overlap': {'t_start': s, 't_end': e}
                    })
                    continue

        # Remise：原始區塊總長 >= THETA，且前 1/3 區段有下半身動作
        if orig_dur >= remisze_theta:
            t_third = orig['t_start'] + (orig_dur / 3.0)
            if has_lower_action(blocks_lower_h, orig['t_start'], t_third):
                tactics.append({'tactic': TACTIC_MAP['Remise'], 't_start': s, 't_end': e})
                continue


        # Riposte：對手剛攻完、間隔小、間隔中我方無攻擊
        orig_start = orig['t_start'] if orig else s
        prev = prev_opp_attack(atk_opp, orig_start, ritpose_gap_tolerate) # prev = ob, 且 ob in opp_atk_blocks
        if prev is not None:
            tactics.append({'tactic': TACTIC_MAP['Riposte'], 't_start': s, 't_end': e})
            continue

        # Attack：預設
        tactics.append({'tactic': TACTIC_MAP['Attack'], 't_start': s, 't_end': e})

    if not tactics:
        return []
    merged = [tactics[0].copy()]
    for t in tactics[1:]:
        prev = merged[-1] # 前一個和現在的比對，如果一樣那就合併再一起
        if t['tactic'] == prev['tactic']:
            prev['t_end'] = t['t_end']
            if 'overlap' in t and 'overlap' in prev:
                prev['overlap']['t_end'] = t['overlap']['t_end']
        else:
            merged.append(t.copy())
    return merged


def process_file(json_path, output_path, remisze_theta=REMISE_THETA, ritpose_gap_tolerate=RITPOSE_GAP_TOLERATE ):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    persons = data.get('persons', [])
    if len(persons) < 2:
        print(f'[tactic]  跳過 (需要兩人，才可以分析互動戰術): {os.path.basename(json_path)}')
        return None

    upper_block = {p['person_id']: p['blocks_upper'] for p in persons} # key, value
    lower_block = {p['person_id']: p['blocks_lower'] for p in persons}
    pids = sorted(upper_block.keys())
    pid_a, pid_b = pids[0], pids[1]

    tactics_a = assign_tactics(upper_block[pid_a], upper_block[pid_b], lower_block[pid_a], remisze_theta, ritpose_gap_tolerate)
    tactics_b = assign_tactics(upper_block[pid_b], upper_block[pid_a], lower_block[pid_b], remisze_theta, ritpose_gap_tolerate)

    result = {
        'video_name': data.get('video_name', ''),
        'fps': data.get('fps', 30.0),
        'total_frames': data.get('total_frames', 0),
        'remisze_theta': remisze_theta,
        'ritpose_gap_tolerate': ritpose_gap_tolerate,
        'persons': [
            {'person_id': pid_a, 'tactics': tactics_a},
            {'person_id': pid_b, 'tactics': tactics_b},
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def run(input_dir=None, remisze_theta=REMISE_THETA, ritpose_gap_tolerate=RITPOSE_GAP_TOLERATE):
    if input_dir is None:
        input_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'outputs_videos', 'filtering')

    output_dir = os.path.join(input_dir, 'tactics')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not files:
        print(f'[tactic] 找不到 json: {input_dir}')
        return

    for jf in files:
        name = os.path.basename(jf)
        print(f'正在處理  {name}')
        process_file(jf, os.path.join(output_dir, name), remisze_theta, ritpose_gap_tolerate)
    print('[tactic] 完成，準備打包所有呈現用資料')
    final_run()


if __name__ == '__main__':
    run(input_dir=None, remisze_theta=REMISE_THETA, ritpose_gap_tolerate=RITPOSE_GAP_TOLERATE)