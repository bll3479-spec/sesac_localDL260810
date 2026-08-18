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
            #(N, 1, 2) int 32
            pts = [(int(p['x']), int(p['y'])) for p in ann ['segmentation']]
            pts = np.array(pts).reshape(-1,1,2)
            #print(pts)

            cv2.fillPoly(mask, [pts], color = class_idx)
            # plt.imshow(mask)
            # plt.show()
        return mask

class NutsDataset(Dataset):
    '''
    Unet 전용 라벨을 return 해주는 NutsDataset -> 라벨의 형태 차이
    Detection -> 'Bounding Box' / Segmentation -> 'mask'가 라벨로 들어감
    '''
    def __init__(self, image_dir, label_dir, file_list, image_size=512, augment=False):
        self.image_dir = image_dir
        self.lable_dir = label_dir
        self.file_list = file_list          #파일 일련번호
        self.image_size = image_size
        self.augment = augment

        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))
        ])
    def __len__(self):
        return len(self.file_list)

    #1. 이미지 return
    #2. 라벨 return -> 라벨 == mask -> mask 또한 '덩어리화'된 숫자 값 
    def __getitem__(self, index):
        filename = self.file_list[index]        #index번째에 해당하는 파일 일련 번호
        image_path = os.path.join(self.image_dir, filename+'.jpg')
        #image를 RGB로 읽어옴, PIL Image
        image = Image.open(image_path).convert('RGB')
        image_w, image_h = image.size           #PIL은 너비 높이

        json_path = os.path.join(self.lable_dir, filename+'.json')
        mask = bulid_mask(json_path=json_path, image_h= image_h, image_w=image_w)        

        #리사이즈 -> 자료형이 다름. image(PIL) / mask(cv2)
        image = image.resize((self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation = cv2.INTER_NEARSET)

        #augmentation
        if self.augment:
            image, mask = self._augment(image,mask)

        #torch의 자료형 변환
        image = self.iamge_transforms(image)
        mask = torch.from_numpy(mask).long()

        return image, mask

if __name__ == '__main__':
    image_path = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\images\798592_594.jpg'
    json_path = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\labels\798592_594.json'

    image = cv2.imread(image_path)
    image_h, image_w = image.shape[:2]
    print(image_h, image_w)
    bulid_mask(json_path=json_path, image_h=image_h, image_w=image_w)