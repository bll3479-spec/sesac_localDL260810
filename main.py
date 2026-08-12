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




if __name__ == '__main__':
#라벨 옮기기

    #1. 원본 소스
    source_dir = r'./Data\peach_dataset\peach_label\yolo_txt_label'
    source_files = [os.path.join(source_dir, x) for x in os.listdir(source_dir)]
    #print(source_files)
    
    #2. train_목록, valid_목록
    train_image_pth = r'./Data\peach_dataset\peach_image\train'
    valid_image_pth = r'./Data\peach_dataset\peach_image\valid'
    train_list = [x for x in os.listdir(train_image_pth)]
    valid_list = [x for x in os.listdir(valid_image_pth)]

    #print(train_list)
    #print(valid_list)
    

    count = 0
    train_target = r'.\Data\peach_dataset\YoloDataset\labels\train'
    valid_target = r'.\Data\peach_dataset\YoloDataset\labels\valid'
    #3.목적지(train/valid) 전송
    for target in source_files:
        f = target.split('\\')[-1].split('.')[0]             #일련번호 생성

        for t in train_list:
            if f in t:
                #print(f'{f} -> train')
                print(target, '-->', os.path.join(train_target, f))
                shutil.copy2(target, os.path.join(train_target, f))
        for v in valid_list:
            if f in v:
                f = f + '.txt'
                shutil.copy2(target, os.path.join(valid_target, f))




#딥러닝 시퀀스
# 1. 데이터 가져옴
# 2. 데이터 정제(preprocessing)
# 3. 알고리즘 선택          ##중요(2번에 영향) 태스크 -> 모델 -> 정제로 형태가 달라짐
# 4. 훈련
# 5. 검증
# 6. 평가
# 7. 배포