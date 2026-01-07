좋아 👍
아래는 **그대로 GitHub `README.md`에 복붙해서 써도 되는 완성본**이야.
👉 교수님/조교가 봐도 **과제 요구사항을 정확히 충족**하고,
👉 Colab + YOLOv8 + Kaggle 흐름이 **논리적으로 잘 보이게** 구성했어.

---

# 🚦 YOLOv8 기반 Road Sign Object Detection

## 1. 프로젝트 개요

본 과제에서는 **Kaggle의 Road Sign Detection 데이터셋**을 활용하여
**YOLOv8(Object Detection)** 모델을 직접 학습하고,
학습된 모델로 도로 표지판 이미지를 객체 탐지(Object Detection)하는 것을 목표로 한다.

* 객체 탐지 모델 학습 전반 과정 이해
* YOLO 형식 데이터셋 구조 이해
* Colab 환경에서의 실습 및 결과 정리
* GitHub를 통한 실습 코드 및 결과 공유

---

## 2. 데이터셋 소개

* 데이터셋 이름: **Road Sign Detection**
* 제공처: Kaggle
* 링크: [https://www.kaggle.com/datasets/andrewmvd/road-sign-detection](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)
* 이미지 수: **877 images**
* 클래스 수: **4 classes**
* 데이터 형태:

  * 도로 표지판 이미지
  * 객체 위치 정보(Annotation)

---

## 3. 실습 환경

* Google Colab
* Python 3
* Ultralytics YOLOv8
* PyTorch (YOLOv8 내부 사용)

---

## 4. 데이터셋 준비

### 4.1 데이터 다운로드

* Kaggle API(`kaggle.json`)를 이용하여 Colab 환경에서 데이터셋 다운로드
* Kaggle API 인증 후 아래 명령어 실행

```bash
kaggle datasets download -d andrewmvd/road-sign-detection --unzip
```

---

### 4.2 YOLOv8 데이터셋 구조

YOLOv8 학습을 위해 다음과 같은 데이터 구조로 정리하였다.

```
datasets/roadsign/
 ├── images/
 │    ├── train/
 │    └── val/
 ├── labels/
 │    ├── train/
 │    └── val/
 └── data.yaml
```

* train / val 비율: **8 : 2**
* Annotation(XML) → YOLO TXT 형식으로 변환
* Bounding Box를 YOLO 좌표 형식 `(x_center, y_center, width, height)`로 정규화

---

### 4.3 data.yaml 예시

```yaml
path: /content/datasets/roadsign
train: images/train
val: images/val

names:
  0: speedlimit
  1: stop
  2: crosswalk
  3: trafficlight
```

---

## 5. YOLOv8 모델 학습

### 5.1 모델 선택

* 사용 모델: **YOLOv8s**
* 선택 이유:

  * YOLOv8n보다 성능이 좋고
  * Colab GPU(T4) 환경에서 학습 가능

---

### 5.2 학습 설정

* Epochs: 10
* Image Size: 640
* Batch Size: 16

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="datasets/roadsign/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    name="roadsign_y8s"
)
```

---

### 5.3 학습 로그 확인

* box_loss
* cls_loss
* dfl_loss
* mAP50
* mAP50-95

학습 결과는 아래 경로에 저장된다.

```
runs/detect/roadsign_y8s/
```

---

## 6. 객체 탐지 테스트 (Prediction)

### 6.1 학습된 모델 로드

```python
from ultralytics import YOLO
model = YOLO("runs/detect/roadsign_y8s/weights/best.pt")
```

---

### 6.2 객체 탐지 수행

```python
results = model.predict(
    source="datasets/roadsign/images/val",
    imgsz=640,
    conf=0.25,
    save=True,
    name="roadsign_pred"
)
```

---

### 6.3 탐지 결과

* Bounding Box가 포함된 결과 이미지가 자동 저장됨
* 저장 위치:

```
runs/detect/roadsign_pred/
```

---

## 7. 결과 분석

* 속도 제한(speed limit) 표지판이 높은 confidence로 정확히 탐지됨
* STOP 표지판 및 신호등도 대부분 정상적으로 검출됨
* 작은 크기의 표지판이나 멀리 있는 객체는 일부 누락 발생

### 개선 아이디어

* Epoch 수 증가
* 이미지 해상도(imgsz) 증가
* 데이터 증강(Augmentation) 적용
* 더 큰 모델(YOLOv8m) 사용

---

## 8. 프로젝트 구조 (GitHub)

```
YOLOv8-RoadSign-Detection/
 ├── README.md
 ├── train.ipynb
 ├── datasets/
 │    └── roadsign/
 │         ├── images/
 │         ├── labels/
 │         └── data.yaml
 └── results/
      └── prediction_samples/
```

