#라벨 만들기 위한 os, json
import os, json
import matplotlib.pyplot as plt
import cv2
import re
import shutil           #.sh / .bash
import torch

# 생성 구조 (Ultralytics 표준):
#   Data/PeachDataset/yolo_dataset/
#     images/train/*.jpg
#     images/valid/*.jpg
#     labels/train/*.txt
#     labels/valid/*.txt


from ultralytics import YOLO
from models.yolo import yolo_train
#실제 코드 augmentation.py -> 호출 main.py
from utils import augmentation as aug
import albumentations as A

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#0818추가 (Unet 관련 모듈) 
from models.unet import Unet, train
from preprocessing.unet_preprocessing import get_dataloader, NUM_CLASSES


def count_params(model):
    #requries_grad: True(훈련시킬 것, 변경 가능) , False(훈련 안 시킴, 변경 불가)
    #p.numel(파라미터의 구성요소 개수)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'전체 파라미터: {total:,}')
    print(f'훈련 가능 파라미터: {trainable:,}')
    print(f'동결 파라미터: {total - trainable:,}')


if __name__ == '__main__':
    import gc

    # 1. Delete model and tensor variables

    # Add any other variables pointing to the GPU

    # 2. Force Python garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    image_dir = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\images'
    label_dir = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\labels'

    train_loader, valid_loader = get_dataloader(image_dir, label_dir, image_size=512, batch_size=2)

    model = Unet(in_channel=3, num_classes= NUM_CLASSES)
    count_params(model=model)

    train(model, train_loader=train_loader, valid_loader=valid_loader, 
          epochs=1, lr=1e-3, save_path=r'./unet_nuts.pth', num_classes=NUM_CLASSES)










    #yolo_train()
    #YOLO 평가
    # source = r'./test_peach.jpg'      #복숭아 이미지
    # model = YOLO('./runs/detect/peach_train01-5/weights/best.pt')
    # model.predict(source = source,
    #               device = 0,
    #               save = True)



    #1.증강용 폴더 생성
    #aug.create_folder(src_folder= r'./Data/peach_dataset/YoloDataset', dst_folder= r'./Data/YoloAugmentation')

    # #2. 증강 랜덤 수행
    # aug.augmentation_image()

    # #3. yolo 라벨 분리
    # aug. load_yolo_label()

    # #4. 3의 결과 저장
    # aug.save_yolo_label()

    #5. (2,3,4 포함) 최종 증강 파이프라인
    #aug.pipe_augmentation()


#딥러닝 시퀀스
# 1. 데이터 가져옴
# 2. 데이터 정제(preprocessing)
# 3. 알고리즘 선택          ##중요(2번에 영향) 태스크 -> 모델 -> 정제로 형태가 달라짐
# 4. 훈련
# 5. 검증
# 6. 평가
# 7. 배포