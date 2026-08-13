# 0810_LT — Deep Learning Training Project

> 분류(Classification) · 객체 탐지(Object Detection) · 세그멘테이션(Segmentation)을 아우르는  
> PyTorch 기반 딥러닝 훈련 파이프라인.  
> VGG, ResNet, YOLOv8, Faster R-CNN, U-Net을 단일 프로젝트에서 실험합니다.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2FCPU-red)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00AAFF)
![Conda](https://img.shields.io/badge/Miniconda-CV-green)

---

## 프로젝트 구조

```
0810_LT/
├── models/                     # 모델 정의
│   ├── model.py                # VGG11 transfer / scratch
│   ├── resnet.py               # ResNet18 / 34 (CIFAR-10 맞춤)
│   ├── faster_rcnn.py          # Faster R-CNN 학습·추론
│   ├── unet.py                 # U-Net + mIoU 평가
│   └── yolo.py                 # YOLOv8 래퍼
│
├── utils/                      # 공통 유틸리티
│   ├── augment.py              # OpenCV 오프라인 증강
│   ├── metrics.py              # mAP · IoU 평가지표
│   ├── yolo_preprocessing.py   # JSON → YOLO txt 변환
│   ├── faster_preprocessing.py # Faster R-CNN 데이터 로더
│   └── visualize.py            # 바운딩박스 시각화
│
├── Data/
│   ├── cifar10_samples/        # 분류용 (10 클래스)
│   └── PeachDataset/
│       ├── peach_json/         # 원본 어노테이션 JSON (100개)
│       ├── peach_image/        # 이미지 train / valid
│       ├── yolo_dataset/       # YOLO 표준 구조 (images/ + labels/)
│       └── yolo_dataset_aug/   # 증강 데이터셋 (train ×3)
│
├── config.py                   # 전역 하이퍼파라미터
├── dataset.py                  # DriveDataset + DataLoader
├── train.py                    # train_one_epoch / evaluate
├── visualize.py                # Loss / Acc 커브 저장
├── main.py                     # 진입점
└── yolo.py                     # YOLO 전처리 + 훈련
```

---

## 모델

| 모델 | 태스크 | 파일 | 설명 |
|---|---|---|---|
| **VGG-11** | 분류 | `models/model.py` | Transfer / Scratch 두 모드 비교. ImageNet 사전학습, features 동결 후 classifier 파인튜닝 |
| **ResNet-18/34** | 분류 | `models/resnet.py` | CIFAR-10 맞춤 커스텀 구현. BasicBlock + skip connection. stride=1, MaxPool 제거 |
| **YOLOv8** | 객체 탐지 | `yolo.py` | Ultralytics 기반. JSON 어노테이션 → YOLO txt 자동 변환. EarlyStopping 내장 |
| **Faster R-CNN** | 객체 탐지 | `models/faster_rcnn.py` | torchvision ResNet50-FPN 백본. 복숭아 1-class 파인튜닝. mAP + IoU 평가 포함 |
| **U-Net** | 세그멘테이션 | `models/unet.py` | 인코더-디코더 구조. mIoU 평가, 클래스별 IoU 집계 |

---

## 데이터셋

| 이름 | 용도 | 구성 | 경로 |
|---|---|---|---|
| CIFAR-10 samples | 분류 | 10 클래스, 클래스별 JPG | `Data/cifar10_samples/` |
| PeachDataset | 객체 탐지 | train 80장 / valid 20장, JSON 어노테이션 | `Data/PeachDataset/` |
| yolo_dataset | YOLO 훈련 | images/ + labels/ + data.yaml | `Data/PeachDataset/yolo_dataset/` |
| yolo_dataset_aug | YOLO 훈련 (증강) | train 320장 (원본 80 + 증강 240) | `Data/PeachDataset/yolo_dataset_aug/` |

---

## 실행 방법

### VGG11 Transfer vs Scratch 비교

```bash
# config.py에서 epochs, lr, modes 설정 후
python main.py
```

### YOLO 전처리 → 증강 → 훈련

```bash
# 1. JSON → YOLO txt + 데이터셋 구조 생성
python utils/yolo_preprocessing.py

# 2. OpenCV 오프라인 증강 (×3배)
python utils/augment.py

# 3. YOLOv8 훈련
python yolo.py
```

### Faster R-CNN

```bash
python models/faster_rcnn.py
```

---

## 주요 설정 (`config.py`)

| 키 | 기본값 | 설명 |
|---|---|---|
| `epochs` | 30 | 최대 에포크 수 |
| `lr` | 1e-3 | Adam 학습률 |
| `batch_size` | 32 | 미니배치 크기 |
| `img_size` | 224 | VGG 입력 해상도 |
| `patience` | 5 | EarlyStopping 기준 에포크 |
| `modes` | `['transfer', 'scratch']` | 실행할 훈련 모드 |
| `num_workers` | 0 | Windows 멀티프로세싱 비활성화 |

---

## 유틸리티

| 파일 | 설명 |
|---|---|
| `utils/augment.py` | OpenCV 기반 오프라인 증강 — 반전, 회전, 이동, 블러, 노이즈, 밝기/대비. 박스 좌표 동시 변환 |
| `utils/metrics.py` | mAP50 / mAP50-95 (torchmetrics), IoU 단일 박스 계산, IoU 배치 행렬 계산 |
| `utils/yolo_preprocessing.py` | JSON 어노테이션 파싱 → YOLO txt 변환 → train/valid 복사 → data.yaml 생성 |
| `utils/faster_preprocessing.py` | Faster R-CNN용 PeachDataset 커스텀 Dataset 클래스 + DataLoader |
| `utils/visualize.py` | 바운딩박스 시각화 — YOLO txt → 픽셀 좌표 역변환 후 사각형 그리기 |

---

## 환경 설정

> **Windows + Miniconda 환경에서는 `num_workers=0` 으로 설정해야 DataLoader 멀티프로세싱 오류가 발생하지 않습니다.**

```bash
conda activate CV

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics torchmetrics opencv-python tqdm matplotlib pyyaml
```
