import json
import random
import os
from smoothing import working

MAMA_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(MAMA_DIR, 'outputs_videos')
os.makedirs(RESULT_DIR, exist_ok=True)

def sith(total_frames=random.randint(15, 200), window_size=21, stride=1, noise_rate=0.40):
    # 生成模仿 PoseC3D dual-head 滑動視窗推論格式的假資料
    segments = []
    person_ids = [1, 2] 
    upper_classes = 2 
    lower_classes = 7 

    for pid in person_ids:
        # 先生成該人物「真實的連續動作」時間軸 
        base_upper = []
        base_lower = []
        
        # 劍法
        while len(base_upper) < total_frames:
            duration = random.randint(15, 30) # 一個動作持續幀數
            label = random.randint(0, upper_classes - 1)
            base_upper.extend([label] * duration)
        base_upper = base_upper[:total_frames]

        # 步法
        while len(base_lower) < total_frames:
            duration = random.randint(10, 30)
            label = random.randint(0, lower_classes - 1)
            base_lower.extend([label] * duration)
        base_lower = base_lower[:total_frames]

        for start in range(0, total_frames - window_size + 1, stride):
            end = start + window_size
            
            mid_point = start + (window_size // 2)
            pred_upper = base_upper[mid_point]
            pred_lower = base_lower[mid_point]

            if random.random() < noise_rate:
                pred_upper = random.randint(0, upper_classes - 1)
            if random.random() < noise_rate:
                pred_lower = random.randint(0, lower_classes - 1)

            score_upper = round(random.uniform(0.55, 0.99), 2)
            score_lower = round(random.uniform(0.55, 0.99), 2)

            segments.append({
                'start_frame': start,
                'end_frame': end,
                'label_upper': pred_upper,
                'label_lower': pred_lower,
                'score_upper': score_upper,
                'score_lower': score_lower,
                'person_id': pid
            })
            
    return segments

if __name__ == "__main__":
    fps = 30.0
    for i in range(1, 11):
        total_frames=random.randint(15, 200)
        video_name = f"test{i:03d}"

        dog_and_cat = sith(
            total_frames=total_frames, 
            window_size=21, 
            stride=1, 
            noise_rate=0.40
        )
        
        save_data = {
            'video_name': f'{video_name}.mp4',
            'fps': fps,
            'total_frames': total_frames,
            'segments': dog_and_cat
        }
        
        output_filename = os.path.join(RESULT_DIR, f'{video_name}.json')
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    print('假資料生成完畢，已啟動自動化投票和聚合')
    working()