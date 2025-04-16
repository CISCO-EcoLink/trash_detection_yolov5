# 🗑️ Trash Detection YOLOv5

Docker 기반으로 구성된 **YOLOv5 객체 검출 시스템**입니다.  
라즈베리파이 5(Raspberry Pi 5)에서 실행되며, 검출된 쓰레기 정보를 **JSON 형식으로 Cisco Splunk로 전송**합니다.

---

## 📌 프로젝트 개요

- **목표**: 쓰레기 적재량을 실시간으로 감지하고, 이를 로그 데이터로 Cisco Splunk에 전달하여 모니터링/분석에 활용
- **기술 스택**:
  - YOLOv5 기반 객체 검출
  - Docker 환경 구성 및 경량화 모델 배포
  - Raspberry Pi 5에서 실시간 추론
  - 검출 결과 JSON으로 가공 후 Cisco Splunk에 전송

---

## 🛠️ 구성 환경

- Python 3.8+
- Docker (Raspberry Pi 5 호환)
- YOLOv5
- OpenCV
- PyTorch (ARM 환경 빌드)
- requests (Splunk 전송용 HTTP 통신)

---

## 🐳 Docker 빌드 및 실행

```bash
# Docker 이미지 빌드
docker build -t trash-detector .

# Docker 컨테이너 실행 (카메라 장치 마운트 예시)
docker run --rm --privileged --device=/dev/video0 -v $(pwd):/app trash-detector



