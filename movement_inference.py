import mmengine
import numpy as np
import os
import glob
import cv2
import json
from mmaction.apis import init_recognizer, inference_recognizer
from PIL import Image, ImageDraw, ImageFont

MAMA_dir = os.path.dirname(os.path.abspath(__file__))
DETECTIONS_DIR = os.path.normpath(
    os.path.join(MAMA_dir, '..', 'ultralytics', 'output_videos', '2d_detections')
)
INPUT_VIDEO_DIR = os.path.normpath(
    os.path.join(MAMA_dir, '..', 'ultralytics', 'output_videos')
)
RESULT_DIR = os.path.join(MAMA_dir, 'outputs_videos')

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

UPPER_LABELS = ['上半身類別0', '上半身類別1']  # 2 類
LOWER_LABELS = ['下半身類別0', '下半身類別1', '下半身類別2', '下半身類別3',
                '下半身類別4', '下半身類別5', '下半身類別6']  # 7 類

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

    for start in range(0, total_frames - window_size + 1, stride):
        end = start + window_size

        clip_kp = person_kp[:, start:end, :, :]
        clip_kps = person_kps[:, start:end, :]

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
    config_path = "configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py"
    checkpoint_path = "configs/skeleton/posec3d/checkpoints/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth"
    # 到時候要換

    print("正在載入模型...")
    model = init_recognizer(config_path, None, device="cuda:0") # None 要改

    video_dirs = sorted([
        d for d in glob.glob(os.path.join(DETECTIONS_DIR, '*'))
        if os.path.isdir(d)
    ])

    if not video_dirs:
        print(f'在 {DETECTIONS_DIR} 下找不到任何影片資料夾')
        return

    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f'找到 {len(video_dirs)} 部影片\n')

    for vdir in video_dirs:
        video_name = os.path.basename(vdir)
        video_path = os.path.join(INPUT_VIDEO_DIR, f'{video_name}.mp4')

        print(f'=== 正在處理: {video_name} ===')

        if not os.path.exists(video_path):
            print(f'  [跳過] 找不到對應影片: {video_path}\n')
            continue

        output_video = os.path.join(RESULT_DIR, f'{video_name}.mp4')
        if os.path.exists(output_video):
            print(f'  [跳過] 輸出影片已存在: {output_video}')
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0:
            fps = 30.0

        # 讀取每個人的骨架，直接從 JSON 載入
        all_results = []
        total_frames = 0
        for pid in [1, 2]:
            person_dir = os.path.join(vdir, str(pid))
            if not os.path.isdir(person_dir):
                print(f'  [跳過] 人物 {pid}（資料夾不存在）')
                continue

            _, total_frames, xys, scores, img_shape = load_person_jsons(person_dir)
            person_kp = xys[np.newaxis, ...]          # (T, 17, 2) -> (1, T, 17, 2)
            person_kps = scores[np.newaxis, ...]      # 一樣 -> (1, T, 17)

            # 檢查資料品質
            empty_count = int(np.sum(np.all(xys == 0, axis=(1, 2))))
            valid_count = total_frames - empty_count
            print(f'  人物 {pid}: 總幀數={total_frames}, 有效={valid_count}, 空幀={empty_count}')

            if np.all(person_kp == 0):
                print(f'  [跳過] 人物 {pid} 所有幀都是 0')
                continue

            # 滑動視窗推論
            results = sliding_window(model, person_kp, person_kps, img_shape, window_size=21, stride=1)

            # 補上 person_id
            for r in results:
                r['person_id'] = pid
            all_results.extend(results)

        if not all_results:
            print(f'  [跳過] 沒有有效片段\n')
            continue

        # 儲存 JSON
        output_json = os.path.join(RESULT_DIR, f'{video_name}.json')
        save_data = {
            'video_name': f'{video_name}.mp4',
            'fps': fps,
            'total_frames': total_frames,
            'segments': all_results
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        # 輸出標註影片
        print(f'  正在輸出標註影片...')
        export_labeled_video(video_path, all_results, fps, total_frames, output_video)
        print(f'  影片已儲存: {output_video}')

    print('全部推論完成！')

if __name__ == '__main__':
    main()