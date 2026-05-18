
import json, os, glob, argparse

MAMA_DIR = os.path.dirname(os.path.abspath(__file__))
#INPUT_DIR = os.path.join(MAMA_DIR, 'outputs_videos', 'final_presentation')

def check_points(blocks, static_start, static_end):
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


def abc(tactics, blocks_upper, blocks_lower):
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


def process_file(filtering_path, tactic_path, output_path):
    with open(filtering_path, 'r', encoding='utf-8') as f:
        fdata = json.load(f)
    with open(tactic_path, 'r', encoding='utf-8') as f:
        tdata = json.load(f)

    block_map = {}
    for p in fdata.get('persons', []):
        block_map[p['person_id']] = {
            'blocks_upper': p['blocks_upper'],
            'blocks_lower': p['blocks_lower'],
        }

    tactic_map = {}
    for p in tdata.get('persons', []):
        tactic_map[p['person_id']] = p['tactics']


    persons_result = []
    for pid in sorted(block_map.keys()): # block 做 foot, epee ; tactic 是戰術
        if pid not in tactic_map:
            continue

        results = abc(
            tactic_map[pid], # tactic_map[person_id] = "tactics": [{"tactic": "Attack", "t_start": 10, "t_end": 10},
            block_map[pid]['blocks_upper'], # "blocks_upper": [{"label": 1,"t_start": 10,"t_end": 14},
            block_map[pid]['blocks_lower'], 
        )

        persons_result.append({
            'person_id': pid,
            'tactic_results': results,
        })

    output = {
        'video_name':    fdata.get('video_name', ''),
        'fps':           fdata.get('fps', 30.0),
        'total_frames':  fdata.get('total_frames', 0),
        'persons':       persons_result,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return output


def final_run(input_dir=None):
    if input_dir is None:
        input_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'outputs_videos', 'filtering')

    tactics_dir = os.path.join(input_dir, 'tactics') # 戰術位置
    output_dir = os.path.join(input_dir, 'final_presentation') # 最終輸出資料夾
    os.makedirs(output_dir, exist_ok=True)

    filtering_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))
    if not filtering_files:
        print(f'[final_package] 找不到 filtering json: {input_dir}')
        return

    for fpath in filtering_files:
        name = os.path.basename(fpath)
        tpath = os.path.join(tactics_dir, name)

        if not os.path.exists(tpath):
            print(f'  skip (無 tactic): {name}')
            continue

        process_file(fpath, tpath, os.path.join(output_dir, name))

    print('[final_package] 全部已完成')


if __name__ == '__main__':
    final_run()