# 베이스 이미지
FROM python:3.9-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /home/rpi4/Ecolink

# YOLOv5 코드 클론
RUN git clone https://github.com/ultralytics/yolov5.git /home/rpi4/Ecolink/yolov5

# yolov5를 Python 모듈 경로에 포함
ENV PYTHONPATH="${PYTHONPATH}:/home/rpi4/Ecolink/yolov5"

# requirements.txt 복사 및 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# OpenCV headless 설치
RUN pip install opencv-python-headless

# 커스텀 모델 및 실행 스크립트 복사
COPY best.pt /home/rpi4/Ecolink/best.pt
COPY detect_webcam.py /home/rpi4/Ecolink/detect_webcam.py

# 실행 명령어
CMD ["python", "/home/rpi4/Ecolink/detect_webcam.py"]
