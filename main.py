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


if __name__ == '__main__':

    #yolo_train()
    #YOLO 평가
    # source = r'./test_peach.jpg'      #복숭아 이미지
    # model = YOLO('./runs/detect/peach_train01-5/weights/best.pt')
    # model.predict(source = source,
    #               device = 0,
    #               save = True)

    #실행 연습
    # fig, ax = plt.subplots(1, 2)
    # #image = cv2.flip(image, 1)
    # image = r'./Data/YoloAugmentation/images/train/A220120XX_10306.jpg'
    # image = cv2.imread(image)
    # print(image.shape)
    # ax[0].imshow(image)
    # image, label = aug.flip_horizontal(image, None)
    # ax[1].imshow(image)
    # plt.show()
    # print(label)

    #1.증강용 폴더 생성
    #aug.create_folder(src_folder= r'./Data/peach_dataset/YoloDataset', dst_folder= r'./Data/YoloAugmentation')

    # #2. 증강 랜덤 수행
    # aug.augmentation_image()

    # #3. yolo 라벨 분리
    # aug. load_yolo_label()

    # #4. 3의 결과 저장
    # aug.save_yolo_label()

    #5. (2,3,4 포함)최종 증강 파이프라인
    #aug.pipe_augmentation()

    #albumentations 적용
    transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
    image = r'./Data/YoloAugmentation/images/train/A220120XX_10306.jpg'

    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]

    plt.imshow(transformed_image)
    plt.show()



#딥러닝 시퀀스
# 1. 데이터 가져옴
# 2. 데이터 정제(preprocessing)
# 3. 알고리즘 선택          ##중요(2번에 영향) 태스크 -> 모델 -> 정제로 형태가 달라짐
# 4. 훈련
# 5. 검증
# 6. 평가
# 7. 배포