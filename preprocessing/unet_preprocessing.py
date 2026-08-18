import os, json, random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

NUTS_CATEGORY_MAP = {
    208: 1,   # 도토리
    209: 2,   # 밤
    210: 3,   # 은행
    211: 4,   # 피칸
    212: 5,   # 호박씨
    213: 6,   # 마카다미아
    214: 7,   # 브라질너트
    215: 8,   # 잣
    216: 9,   # 호두
    217: 10,  # 해바라기씨
    218: 11,  # 밤송이
    219: 12,  # 아몬드
    220: 13,  # 피스타치오
    221: 14,  # 땅콩
    222: 15,  # 캐슈넛
}
CLASS_NAMES = {
    0: '배경',
    1: '도토리', 2: '밤', 3: '은행', 4: '피칸', 5: '호박씨',
    6: '마카다미아', 7: '브라질너트', 8: '잣', 9: '호두', 10: '해바라기씨',
    11: '밤송이', 12: '아몬드', 13: '피스타치오', 14: '땅콩', 15: '캐슈넛',
}
NUM_CLASSES = 16 

def bulid_mask(json_path, image_h, image_w):
    '''
    특정한 json 파일의 polygon을 읽어서 마스크 생성
    '''
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        mask = np.zeros((image_h, image_w), dtype = np.uint8)

        for ann in data ['annotations']:
            #딕셔너리.get(키, 값): 키에 해당하는 값을 가져와 / 값: 디폴트값. 해당하는 키-값쌍이 없다면 디폴트 값(0:배경)
            class_idx=NUTS_CATEGORY_MAP.get(ann['category_id'], 0)         #0: 배경 없음

class NutsDataset(Dataset):
    '''
    Unet 전용 라벨을 return 해주는 NutsDataset -> 라벨의 형태 차이
    Detection -> 'Bounding Box' / Segmentation -> 'mask'가 라벨로 들어감
    '''
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, index):
        pass


if __name__ == '__main__':
    image_path = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\images\798592_594.jpg'
    json_path = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\labels\798592_594.json'

    image = cv2.imread(image_path)
    image_h, image_w = image.shape[:2]
    print(image_h, image_w)
    bulid_mask(json_path=json_path, image_h=image_h, image_w=image_w)