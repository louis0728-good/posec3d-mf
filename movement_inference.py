import mmengine
import sys
import argparse
import torch
import numpy as np
import os
import glob
import cv2
import json
from tqdm import tqdm
from mmaction.apis import init_recognizer, inference_recognizer
from PIL import Image, ImageDraw, ImageFont

_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

KEYPOINT_BASE = r"D:\project\2d_output_videos"     # ViTPose 輸出的 2D 關鍵點 JSON 根目錄
INPUT_VIDEO_BASE = r"D:\project\clipped"          # 原始影片根目錄
RESULT_BASE = r"D:\project\action_output_videos"  # 動作辨識輸出根目錄 (會自動保留資料夾結構)

def parse_args():
    ap = argparse.ArgumentParser(description='PoseC3D 動作辨識')
    ap.add_argument('--keypoint_base',    default=KEYPOINT_BASE)
    ap.add_argument('--input_video_base', default=INPUT_VIDEO_BASE)
    ap.add_argument('--result_base',      default=RESULT_BASE)
    ap.add_argument('--match', default=None,
                    help='只處理這一場（子資料夾名），不給就全部')
    ap.add_argument('--config',
        default="configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py")
    ap.add_argument('--checkpoint',
        default="checkpoints/best.pth")
    return ap.parse_args()


# 每個人的顏色
PERSON_COLORS = {
        1: (0, 255, 0),    # 綠色
        2: (0, 165, 255),  # 橘色
    }

# NTU RGB+D 60 動作標籤
NTU60_LABELS = [
        '喝水', '吃飯', '刷牙', '梳頭髮', '丟東西',
        '撿東西', '扔東西', '坐下', '站起來', '拍手',
        '閱讀', '寫字', '撕紙', '穿外套', '脫外套',
        '穿鞋', '脫鞋', '戴眼鏡', '脫眼鏡',
        '戴帽子', '脫帽子', '歡呼', '揮手',
        '踢東西', '手伸進口袋', '單腳跳', '跳躍',
        '打電話', '滑手機', '打鍵盤',
        '指向某物', '自拍', '看手錶',
        '搓手', '點頭/鞠躬', '搖頭', '擦臉',
        '敬禮', '雙手合十', '雙手交叉胸前',
        '打噴嚏/咳嗽', '搖搖晃晃', '跌倒', '摸頭',
        '摸胸口', '摸背', '摸脖子', '噁心/嘔吐',
        '搧扇子', '打/甩對方', '踢對方',
        '推對方', '拍對方的背',
        '用手指指對方', '擁抱對方',
        '拿東西給對方', '碰對方口袋',
        '握手', '走向彼此', '離開彼此'
    ]

UPPER_LABELS = ['無', '直刺']
LOWER_LABELS = ['無', '前進', '後退', '長刺', '飛刺', ' 前進長刺'] 

def load_person_jsons(json_dir): # 這裡先把所以人的 json 都累積起來
    # 讀取某一個人資料夾裡所有 .json，回傳 (video_name, total_frames, xy, score)
    files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    if not files:
        raise FileNotFoundError(f'找不到任何 json: {json_dir}')

    #frames = []
    xys = []
    scores = []
    video_name = None
    img_w = None  
    img_h = None

    for ff in files:
        with open(ff, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if video_name is None:
            video_name = data.get('video_name', os.path.basename(json_dir))
            img_w = data.get('width')
            img_h = data.get('height')

        #frame_idx = int(data['frame'])
        kps = data.get('keypoints', [])
        # 沒偵測到人 / 空關鍵點：直接補 0
        if len(kps) == 0:
            xy = np.zeros((17, 2), dtype=np.float32)
            sc = np.zeros((17,), dtype=np.float32)
        else:
            #kps = sorted(kps, key=lambda x: x['id'])

            xy = np.array([[kp['x'], kp['y']] for kp in kps], dtype=np.float32)
            sc = np.array([kp['v'] for kp in kps], dtype=np.float32)

            if xy.shape != (17, 2):
                raise ValueError(f'{ff} keypoint shape 是 {xy.shape}, 與預期不符 (17, 2)')
            if sc.shape != (17,):
                raise ValueError(f'{ff} score shape 是 {sc.shape}, 與預期不符 (17,)')

        #frames.append(frame_idx)
        xys.append(xy)
        scores.append(sc)

    # 依 frame 排序後，補成連續時間軸
    #order = np.argsort(frames)
    #frames = np.array(frames)[order]
    # 排序後應該怎麼重排 frames 照舊順序，但實際讀取資料看 order 順序，不影響原本 frames 的排版
    xys = np.stack(xys, axis=0)    # (Tp, 17, 2)
    scores = np.stack(scores, axis=0) # (Tp, 17)
    total_frames = len(xys)

    return video_name, total_frames, xys, scores, (img_h, img_w)

def sliding_window(model, person_kp, person_kps, img_shape, window_size=21, stride=1):
    """
    對單一人物做滑動視窗推論。
    Args:
        person_kp:  (1, T, 17, 2)
        person_kps: (1, T, 17)
        img_shape:  (h, w)
        window_size: 對齊 pipeline 的 clip_len=21
    """
    total_frames = person_kp.shape[1]
    results = []

    for start in tqdm(range(0, total_frames - window_size + 1, stride),
                      desc='  推論中', leave=False):
        end = start + window_size

        clip_kp = person_kp[:, start:end, :, :]
        clip_kps = person_kps[:, start:end, :]

        # 有效幀檢查與前向填補 (防呆)
        valid = ~np.all(clip_kp == 0, axis=(2, 3))[0]     # (21,) 每幀是否有偵測
        if valid.sum() < 15:                              # 有效幀不足就跳過
            continue
            
        clip_kp = clip_kp.copy()
        clip_kps = clip_kps.copy()
        last = None
        for t in range(clip_kp.shape[1]):
            if valid[t]:
                last = (clip_kp[:, t].copy(), clip_kps[:, t].copy())
            elif last is not None:
                clip_kp[:, t], clip_kps[:, t] = last

        if np.all(clip_kp == 0):
            continue

        data_dict = {
            'keypoint': clip_kp,
            'keypoint_score': clip_kps,
            'img_shape': img_shape,
            'total_frames': window_size,
            'start_index': 0,
            'modality': 'Pose'
        }

        result = inference_recognizer(model, data_dict)

        # dual head 回傳的是 pred_score_upper 和 pred_score_lower
        score_upper = result.pred_score_upper.cpu().numpy()
        score_lower = result.pred_score_lower.cpu().numpy()

        pred_upper = int(np.argmax(score_upper))
        pred_lower = int(np.argmax(score_lower))

        results.append({
            'start_frame': start,
            'end_frame': end,
            'label_upper': pred_upper,
            'label_lower': pred_lower,
            'score_upper': round(float(score_upper[pred_upper]), 2),
            'score_lower': round(float(score_lower[pred_lower]), 2),
        })

    return results

def chinese(frame, text, position, font_path, font_size, color):

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def export_labeled_video(video_path, results, fps, total_frames, output_path):
    #                    原始影片路徑  
    """
    把動作標籤疊到原始影片上，輸出新影片
    """
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    """  如果同一幀被多個重疊視窗覆蓋，就用分數最高的那段片段。  """

    person_ids = sorted(set(r['person_id'] for r in results)) # 排序 ID: 1, 2 
    frame_labels = {pid: [None] * total_frames for pid in person_ids}
    """
    建立一個 frame_labels  大字典
    {
        1: [None, None, None, ..., None]
        2:[...]
    }
    """
    for r in results: # 同部影片不同人
        pid = r['person_id']
        for i in range(r['start_frame'], min(r['end_frame'], total_frames)): # 這個視窗覆蓋了哪些幀，逐幀處理
            avg_score = (r['score_upper'] + r['score_lower']) / 2
            if frame_labels[pid][i] is None or avg_score > (frame_labels[pid][i]['score_upper'] + frame_labels[pid][i]['score_lower']) / 2:
            # 因為 sliding window 有重疊，所以同一幀可能同時屬於多個片段。這裡的策略是：保留分數最高的片段結果
                frame_labels[pid][i] = r

    frame_idx = 0 # 目前處理第幾幀
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        y_offset = 10 # 文字的 y 座標
        font_path = r'C:\Windows\Fonts\msjh.ttc'  # 微軟正黑體，Windows 都有
        font_size = 28

        for pid in person_ids:
            if frame_idx < len(frame_labels[pid]) and frame_labels[pid][frame_idx] is not None: # 確認確實有資訊
                info = frame_labels[pid][frame_idx]
                upper_id = info['label_upper']
                lower_id = info['label_lower']
                upper_text = UPPER_LABELS[upper_id] if upper_id < len(UPPER_LABELS) else f'upper_{upper_id}'
                lower_text = LOWER_LABELS[lower_id] if lower_id < len(LOWER_LABELS) else f'lower_{lower_id}'
                display = f'ID {pid}: 上:{upper_text}({info["score_upper"]:.2f}) 下:{lower_text}({info["score_lower"]:.2f})'

                color_bgr = PERSON_COLORS.get(pid, (255, 255, 255))
                # PIL 用 RGB，OpenCV 用 BGR，要反轉
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

                # 畫黑底
                text_w = len(display) * font_size  # 粗估寬度
                cv2.rectangle(frame, (10, y_offset), (20 + text_w, y_offset + font_size + 10), (0, 0, 0), -1)

                # 畫中文
                frame = chinese(frame, display, (15, y_offset + 3), font_path, font_size, color_rgb)

                y_offset += font_size + 15

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def main():
    args = parse_args()
    # 到時候要換

    print("正在載入模型...")
    model = init_recognizer(args.config, args.checkpoint, device="cuda:0")

    # 取得 2D 骨架底下的所有子資料夾 (ex: 1-8(1)-1)
    sub_dirs = [d for d in os.listdir(args.keypoint_base)
                if os.path.isdir(os.path.join(args.keypoint_base, d))]

    if args.match:
        sub_dirs = [d for d in sub_dirs if d == args.match]

    if not sub_dirs:
        print(f'在 {args.keypoint_base} 下找不到任何子資料夾')
        return

    failed = []
    print(f'找到 {len(sub_dirs)} 個子資料夾等待處理\n')

    # 第一層：遍歷子資料夾 (ex: 1-8(1)-1)
    for sub_dir in sub_dirs:
        sub_keypoint_dir = os.path.join(args.keypoint_base, sub_dir)
        video_dirs = sorted([d for d in glob.glob(os.path.join(sub_keypoint_dir, '*')) if os.path.isdir(d)])

        if not video_dirs:
            continue

        print(f'=== 正在處理子資料夾: {sub_dir} (共 {len(video_dirs)} 部影片) ===')

        # 第二層：遍歷子資料夾內的各影片骨架目錄 (ex: test001)
        for vdir in video_dirs:
            video_name = os.path.basename(vdir)
            try:
                video_path = os.path.join(args.input_video_base, sub_dir, f'{video_name}.mp4')

                print(f'--- 處理影片: {video_name} ---')

                if not os.path.exists(video_path):
                    print(f'  [跳過] 找不到對應原始影片: {video_path}\n')
                    continue

                # 為每部影片建立專屬輸出資料夾 (ex: D:\project\action_output_videos\1-8(1)-1\test001\)
                video_result_dir = os.path.join(args.result_base, sub_dir, video_name)
                os.makedirs(video_result_dir, exist_ok=True)

                output_video = os.path.join(video_result_dir, f'{video_name}_action.mp4')
                output_json = os.path.join(video_result_dir, f'{video_name}_action.json')

                if os.path.exists(output_video) and os.path.exists(output_json):
                    print(f'  [跳過] 輸出已存在: {video_name}\n')
                    continue

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps == 0 or np.isnan(fps):
                    fps = 30.0

                # 讀取每個人的骨架 JSON
                all_results = []
                total_frames = 0
                for pid in [1, 2]:
                    person_dir = os.path.join(vdir, str(pid))
                    if not os.path.isdir(person_dir):
                        print(f'  [跳過] 人物 {pid}（資料夾不存在）')
                        continue

                    _, total_frames, xys, scores, img_shape = load_person_jsons(person_dir)
                    person_kp = xys[np.newaxis, ...]
                    person_kps = scores[np.newaxis, ...]

                    if np.all(person_kp == 0):
                        print(f'  [跳過] 人物 {pid} 所有幀都是 0')
                        continue

                    # 滑動視窗推論
                    results = sliding_window(model, person_kp, person_kps, img_shape, window_size=21, stride=1)

                    for r in results:
                        r['person_id'] = pid
                    all_results.extend(results)

                if not all_results:
                    print(f'  [跳過] 沒有有效片段\n')
                    continue

                # 儲存獨立 JSON
                save_data = {
                    'video_name': f'{video_name}.mp4',
                    'fps': fps,
                    'total_frames': total_frames,
                    'segments': all_results
                }
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)

                # 輸出專屬影片
                print(f'  正在輸出標註影片...')
                export_labeled_video(video_path, all_results, fps, total_frames, output_video)
                print(f'  影片已儲存: {output_video}\n')

            except Exception as e:
                print(f'  [失敗] {sub_dir}/{video_name}: {type(e).__name__}: {e}\n')
                failed.append(f'{sub_dir}/{video_name}')
                continue

    if failed:
        print(f'\n完成，但有 {len(failed)} 部影片失敗:')
        for f in failed:
            print(f'  - {f}')
        sys.exit(1)
    print('全部子資料夾動作推論完成！')

if __name__ == '__main__':
    main()