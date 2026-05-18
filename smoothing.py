import json
import os
import glob
import argparse
from collections import Counter
from tactic_labeling import run

MAMA_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(MAMA_DIR, 'outputs_videos')
SMOOTH_N = 10
WINDOW_SIZE = 21
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

def flatten(segments, total_frames, window_size=21):
    # 邊界幀（無中間幀覆蓋）留 None。

    mid = window_size // 2
    persons = sorted(set(seg['person_id'] for seg in segments))
    result = {}

    for pid in persons:
        upper = [None] * total_frames
        lower = [None] * total_frames

        pid_segs = [s for s in segments if s['person_id'] == pid] # 1, 2
        for seg in pid_segs:
            acting = seg['start_frame'] + mid # 我是這樣想，視窗 21 取中間帧當作代表，所以第一個 0~ 20 代表帧會是 10
            if 0 <= acting < total_frames:
                upper[acting] = seg['label_upper']
                lower[acting] = seg['label_lower']

        result[pid] = {'upper': upper, 'lower': lower}

    return result


def mj_vote(seq, n_times):
    """
    窗口大小 3（t-1, t, t+1）的多數決投票，重複 N 次。
    - None 的幀不參與投票，也不被修改。
    - 平票時保持原值 t 不變。
    """
    arr = list(seq)

    for _ in range(n_times):
        new_arr = list(arr)
        for t in range(len(arr)):
            if arr[t] is None:
                continue

            # 沒有 none 且 應該三
            smooth_window = []
            for dt in [-1, 0, 1]:
                idx = t + dt
                if 0 <= idx < len(arr) and arr[idx] is not None:
                    smooth_window.append(arr[idx])

            if not smooth_window:
                continue

            counter = Counter(smooth_window)
            most_common = counter.most_common() # [(x, a 次), (y, b 次)] a 應該會 >= b

            # 檢查是否有唯一最大值（非平票）
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                new_arr[t] = most_common[0][0]

        arr = new_arr

    return arr


def we_are_freinds(seq):
    """
    將連續且標籤一致的幀聚合為區塊。
    跳過 None 幀。
    回傳 list {label, t_start, t_end}
    """
    blocks = []
    current_label = None
    t_start = None

    for t, label in enumerate(seq):
        if label is None:
            # 遇到 None 結束
            if current_label is not None:
                blocks.append({
                    'label': current_label,
                    't_start': t_start,
                    't_end': t - 1
                })
                current_label = None
                t_start = None
            continue

        if label == current_label:
            continue  # 延續同一段
        else:
            # 新標籤 先結束前一段
            if current_label is not None:
                blocks.append({
                    'label': current_label,
                    't_start': t_start,
                    't_end': t - 1
                })
            current_label = label
            t_start = t

    if current_label is not None:
        blocks.append({
            'label': current_label,
            't_start': t_start,
            't_end': len(seq) - 1
        })

    return blocks


def working(input_dir=None, smooth_n=SMOOTH_N, window_size=WINDOW_SIZE):

    if input_dir is None:
        input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs_videos')

    output_dir = os.path.join(input_dir, 'filtering')
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))

    if not json_files:
        print(f'[working] 在 {input_dir} 找不到任何 json 檔案。(from, fi....AND...p....py)')
        return

    for jf in json_files:
        basename = os.path.basename(jf)
        output_path = os.path.join(output_dir, basename)
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_frames = data['total_frames']
        segments = data['segments']
        fps = data['fps']
        video_name = data['video_name']
        per_person = flatten(segments, total_frames, window_size)
        all_person_results = []

        for pid in sorted(per_person.keys()):
            raw_upper = per_person[pid]['upper']
            raw_lower = per_person[pid]['lower']

            smoothed_upper = mj_vote(raw_upper, smooth_n)
            smoothed_lower = mj_vote(raw_lower, smooth_n)

            blocks_upper = we_are_freinds(smoothed_upper)
            blocks_lower = we_are_freinds(smoothed_lower)

            all_person_results.append({
                'person_id': pid,
                'blocks_upper': blocks_upper,
                'blocks_lower': blocks_lower,
                'raw_upper': raw_upper,
                'raw_lower': raw_lower,
                'smoothed_upper': smoothed_upper,
                'smoothed_lower': smoothed_lower,
            })

        output_data = {
            'video_name': video_name,
            'fps': fps,
            'total_frames': total_frames,
            'window_size': window_size,
            'smooth_n': smooth_n,
            'persons': all_person_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    print('也都濾波、聚合好了，準備開始判斷戰術')
    run()

if __name__ == '__main__':

    working(
        input_dir=INPUT_DIR,
        smooth_n=SMOOTH_N,
        window_size=WINDOW_SIZE
    )
