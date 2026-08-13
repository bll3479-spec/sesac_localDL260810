# sesac_localDL260810

복숭아(peach) 이미지에 대한 객체 탐지(Object Detection) 프로젝트. YOLO를 메인으로 사용하고 있고, Faster R-CNN도 실험 중. `PreVersion/`에는 CIFAR-10 이미지 분류 실험으로 시작했던 이전 버전 코드가 남아 있음.

## 실행 환경
- Windows, PowerShell 기본 / Bash(Git Bash)도 사용 가능.
- `requirements.txt`에는 `torch`, `matplotlib` 등은 있지만 `opencv-python`(cv2), `ultralytics`, `albumentations`가 빠져 있음 — 실제 사용 중인 가상환경(예: `C:\ProgramData\anaconda3\envs\CV`)에는 설치돼 있는 것으로 보이나, `requirements.txt`와 실제 설치 패키지가 어긋나 있으니 새 환경 세팅 시 주의.
- 코드 내 상대경로(`./Data/...`)가 많아서 **반드시 프로젝트 루트에서 실행**해야 함.

## 디렉토리 구조 (핵심)
```
Data/
  cifar10_samples/        # CIFAR-10 분류용, airplane~truck 10개 클래스 폴더 (하위에 이미지 직접 있음)
  peach_dataset/
    YoloDataset/           # YOLO 학습 원본 (images/, labels/ - train/valid)
    peach_image/, peach_label/
  YoloAugmentation/        # augmentation.py가 만드는 증강 작업 사본 (원본 복사 후 여기서 증강)

models/
  yolo.py           # yolo_train() - Ultralytics YOLO 학습 (yolo_setting.yaml 사용, class: peach 1종)
  vgg.py            # get_vgg_model() - vgg11 전이학습 (classifier만 학습)
  resnet_pre.py     # get_resnet_model() - torchvision resnet34 전이학습 (fc만 학습)
  resnet.py         # ResNet 직접 구현 (BasicBlock 기반, 학습용 커스텀 구현)

utils/
  augmentation.py   # YOLO 데이터셋 증강 파이프라인 (아래 "알려진 이슈" 참고)
  DataLoader.py     # DriveDataset(Dataset) - 폴더명=클래스 구조의 이미지 분류 데이터셋 + get_dataloader()
  graph.py          # draw_plot(history, save_path) - train/valid loss·acc 곡선을 이미지로 저장
  visualize.py      # show_yolo_label(image_file, txt_file) - YOLO 라벨을 이미지 위에 박스로 그려 확인

train.py            # train_one_epoch() - 분류 모델용 1 에포크 학습 루프
eval.py             # evaluate() - 분류 모델용 검증 루프
main.py             # 현재 작업 중인 실험 스크립트 (자주 바뀜, 아래 참고)
PreVersion/old_main.py  # CIFAR-10 분류 실험의 예전 메인 스크립트
yolo_setting.yaml   # Ultralytics YOLO data yaml (path/train/val/names)
```

## 데이터 흐름 연결 (old_main.py 기준, 분류 실험)
```
utils/DataLoader.py  get_dataloader()  →  train_loader, valid_loader
models/resnet_pre.py get_resnet_model() →  전이학습 모델
train.py / eval.py                     →  에포크별 학습/검증
utils/graph.py        draw_plot()      →  result.jpg 저장
```
`old_main.py`의 `run_epoch()`가 위 요소들을 묶어서 EPOCHS만큼 반복.

## main.py 현재 상태
계속 실험적으로 바뀌는 스크립트. 현재는 Faster R-CNN(torchvision) 모델을 로드해서 구조/파라미터 수 확인하는 코드가 있고, YOLO 학습/증강 파이프라인 호출부는 대부분 주석 처리되어 있음. `count_params()`로 전체/훈련가능/동결 파라미터 수를 출력.

## 알려진 이슈 / TODO
- **utils/augmentation.py `pipe_augmentation()`**
  - 106행 `break`가 이미지 파일 순회(for) 최상단에 있어서, 전체 데이터셋 중 **첫 번째 이미지 1장만 증강**되고 나머지는 처리되지 않음.
  - 104행에서 증강 라벨을 `out_name.txt`가 아니라 원본 `filename.txt`로 저장 — 증강 이미지(`_0`, `_1`, `_2`)에 대응하는 라벨 파일이 없고, 원본 라벨 파일이 마지막 증강 결과로 덮어써짐.
  - `augmentation_image()`에는 flip_horizontal/vertical만 구현되어 있고 rotate/translate/blur/noise/brightness/contrast는 주석 처리된 미구현 상태.
  - `applay_albumentation()`(함수명 오타, apply가 맞음)은 albumentations 데모용 독립 함수. `RandomCrop`을 쓰면서도 라벨(bbox) 좌표는 갱신하지 않아 YOLO 라벨과 매칭 안 됨.
- **utils/DataLoader.py `get_dataloader()`**
  - `root_dir`이 `Data` 폴더 전체로 지정되어 있음. `DriveDataset`은 "root_dir 바로 아래 폴더=클래스, 그 안에 이미지가 직접 있음" 구조를 기대하는데, 실제로 `Data` 바로 아래(`cifar10_samples`, `peach_dataset`, `YoloAugmentation`)는 전부 하위에 또 폴더만 있어서 이미지 파일이 하나도 안 걸림 → `self.image_path`가 빈 리스트 → `DriveDataset.__init__` 36행 `print(f'{self.image_path[0]}...')`에서 `IndexError` 발생.
  - CIFAR-10 클래스 구조(`airplane`~`truck` 10개)와 정확히 일치하는 건 `Data/cifar10_samples`이므로, 원래 의도는 `root_dir = Data/cifar10_samples`였을 것으로 추정됨. 고치지 않은 상태.
- **PreVersion/old_main.py**
  - `run_epoch()`(35~66행) 내부에서 `train_loader`, `valid_loader`를 파라미터로 받지 않고 전역 변수처럼 그냥 사용 — `__main__` 블록에서 먼저 할당된 뒤 호출되는 순서라 지금은 우연히 동작하지만 재사용 시 `NameError` 위험.
  - 63, 65행에서 이미 1부터 시작하는 에포크 번호 `e`에 `+1`을 추가로 더해 저장/출력 — 실제보다 한 에포크 밀려서 기록됨 (예: 1에포크 결과가 "2 epoch_...pth"로 저장).
- 지금까지 세션에서 고친 것: `main.py`의 `FasterRCNN_Resnet50_FPN_Weights` → `FasterRCNN_ResNet50_FPN_Weights` 오타, `.DEFALUT` → `.DEFAULT` 오타 수정 완료.
