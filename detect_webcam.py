import torch
import cv2
import time
import sys
import numpy as np
import os
import json
from datetime import datetime
import pytz

# yolov5 경로 추가
sys.path.append(os.path.dirname(__file__))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

# 모델 설정
device = 'cpu'
model = DetectMultiBackend('/home/rpi4/Ecolink/best.pt', device=device)
stride, names, pt = model.stride, model.names, model.pt
img_size = 640

# 디렉토리 설정
base_dir = "/home/rpi4/Ecolink"
images_dir = os.path.join(base_dir, "images")
jsons_dir = os.path.join(base_dir, "jsons")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(jsons_dir, exist_ok=True)

# 시간대 설정
kst = pytz.timezone('Asia/Seoul')
frame_count = 0

# 비디오 캡처 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    now = datetime.now(kst)

    ret, frame = cap.read()
    if not ret:
        break

        # ✅ 00초 / 30초가 아닐 경우 스킵
    if now.second not in [0, 30]:
        time.sleep(0.5)
        continue

    total_area = 0

    img = letterbox(frame, (img_size, img_size), stride=stride)[0]
    if img is None or len(img.shape) != 3:
        print(f"Invalid image shape: {img.shape}")
        continue

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                width = x2 - x1
                height = y2 - y1
                area = width * height
                total_area += area

                label = f'{names[int(cls)]} {conf:.2f} Area:{area}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"[{now.strftime('%H:%M:%S')}] Total detected area: {total_area} pixels")

    # 파일 이름용 시간 문자열
    filename_base = now.strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(images_dir, f"output_{filename_base}.jpg")
    json_path = os.path.join(jsons_dir, f"output_{filename_base}.json")

    success = cv2.imwrite(image_path, frame)
    if success:
        print(f"✅ Saved frame to: {image_path}")

        frame_area = frame.shape[0] * frame.shape[1]
        trash_level = round((total_area / frame_area) * 100 * 4, 2)
        if trash_level > 100:
            trash_level = 100.0

        json_data = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "location": "gangnam",
            "hazardous_gas": "danger",
            "temperature": 27.4,
            "humidity": 58,
            "trash_level": trash_level
        }

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"✅ Saved metadata to: {json_path}")

        # ✅ 오래된 파일 삭제 (5쌍 초과 시)
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.startswith("output_") and f.endswith(".jpg")
        ])
        json_files = sorted([
            f for f in os.listdir(jsons_dir)
            if f.startswith("output_") and f.endswith(".json")
        ])

        num_pairs = min(len(image_files), len(json_files))
        if num_pairs > 5:
            num_to_delete = num_pairs - 5
            for i in range(num_to_delete):
                old_image = os.path.join(images_dir, image_files[i])
                old_json = os.path.join(jsons_dir, json_files[i])
                for path in [old_image, old_json]:
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"🗑️ Deleted old file: {path}")
    else:
        print(f"❌ Failed to save image to: {image_path}")

    frame_count += 1
    time.sleep(1.0)  # 1초 대기 (같은 초 중복 처리 방지)

cap.release()
