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


if __name__ == '__main__':

    # train = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\peach_dataset\YoloDataset\labels\valid'
    # paths = [os.path.join(train, x) for x in os.listdir(train)]

    # for i in paths:
    #     t = i.split('\\')[-1] + '.txt'
    #     os.rename(i, os.path.join(train, t))


    #YOLO 라이브러리 세팅(pip)
    yaml_path = r'./yolo_setting.yaml'
    #YOLO 훈련
    result = YOLO('yolov8n.pt').train(
                data=yaml_path,  
                epochs = 50, 
                imgsz = 640, 
                batch=16, 
                save = True,
                device = 0, 
                plots = True,
                name = 'peach_train01')
    print('훈련 완료')

    #YOLO 평가



#딥러닝 시퀀스
# 1. 데이터 가져옴
# 2. 데이터 정제(preprocessing)
# 3. 알고리즘 선택          ##중요(2번에 영향) 태스크 -> 모델 -> 정제로 형태가 달라짐
# 4. 훈련
# 5. 검증
# 6. 평가
# 7. 배포