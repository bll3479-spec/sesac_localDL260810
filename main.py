#라벨 만들기 위한 os, json
import os, json
import matplotlib.pyplot as plt
import cv2
import re
import shutil           #.sh / .bash

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

def count_params(model):
    #requries_grad: True(훈련시킬 것, 변경 가능) , False(훈련 안 시킴, 변경 불가)
    #p.numel(파라미터의 구성요소 개수)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'전체 파라미터: {total:,}')
    print(f'훈련 가능 파라미터: {trainable:,}')
    print(f'동결 파라미터: {total - trainable:,}')


if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    #Faster RCNN
    #Backbone => 이미지 특징 추출 => 저수준 특징 추출(ResNet)
    #RPN(Resion Proposal Networks) => Bounding Box의 후보 제안
    #ROI Head => RPN을 보고 분류 수행, bbox의 사이즈 보정**
    #Faster RCNN : 분류하고자 하는 객체의 개수 (+1 해야함, 얘는 배경까지(2-stage 모델이니까))
    print(model)
    
    #ROI Head 변경
    in_feautures = model.roi_heads.box_predictor.cls_score.in_features      #(모델 피쳐값 임시저장용)

    #ROI Head도 2개의 부속품이 있음 -> Cls_score(분류) / Bbox_predictor(바운딩 박스 찾기)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feautures, num_classes = 1+1)
    count_params(model)




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