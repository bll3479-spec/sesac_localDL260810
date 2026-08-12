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

if __name__ == '__main__':

    yolo_train()
    #YOLO 평가
    # source = r'./test_peach.jpg'      #복숭아 이미지
    # model = YOLO('./runs/detect/peach_train01-5/weights/best.pt')
    # model.predict(source = source,
    #               device = 0,
    #               save = True)


#딥러닝 시퀀스
# 1. 데이터 가져옴
# 2. 데이터 정제(preprocessing)
# 3. 알고리즘 선택          ##중요(2번에 영향) 태스크 -> 모델 -> 정제로 형태가 달라짐
# 4. 훈련
# 5. 검증
# 6. 평가
# 7. 배포