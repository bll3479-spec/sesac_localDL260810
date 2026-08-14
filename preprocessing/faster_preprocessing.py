import os, json, random, shutil
from PIL import Image

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

NUTS_CLASS_NAMES = {v: k_name for k_name, v in {
    '도토리': 1, '밤': 2, '은행': 3, '피칸': 4, '호박씨': 5,
    '마카다미아': 6, '브라질너트': 7, '잣': 8, '호두': 9, '해바라기씨': 10,
    '밤송이': 11, '아몬드': 12, '피스타치오': 13, '땅콩': 14, '캐슈넛': 15,
}.items()}



#@@모델 먼저 고르고 -> 맞는 전처리 생각하기: 모델 구조를 보고 어떤 라벨을 원하는지 알 것@@
#pytorch model zoo, faster rcnn: 
class NutDataset(Dataset):
    #데이터셋을 만들 때 필요한 핵심 정보: 이미지, 라벨 폴더의 경로 / transform
    #transforms=None default 설정
    def __init__(self, image_dir, label_dir, transforms = None):
        self.image_dir = image_dir
        self.transform = transforms
        self.label_dir = label_dir
        #
        self.samples = []
        self.extract_label_data(self.label_dir)     #json에서 이미지와 라벨의 한 쌍을 만들어주려고.

    #데이터셋의 길이? -> 데이터가 있는 위치의 파일 개수 반환 -> 한 개의 이미지에 여러 개의 라벨 존재 가능.
    # 라벨의 개수를 데이터 개수라고 보는 것이 맞음.
    def __len__(self):
        return len(self.samples)
    
    #이미지와 라벨(target)의 쌍 반환
    def __getitem__(self, index):
        #self.samples 안에 이미지, 라벨 모두 존재
        sample = self.samples[index]
        #이미지
        image = Image.open(sample['img_path']).convert('RGB')
        #라벨 -> bbox, cls(어떤 오브젝트인지)
        boxes = torch.tensor(sample['boxes'], dtype=torch.float32)
        cls = torch.tensor(sample['labels'], dtype=torch.int64)

                # 0, 1 ,2 ,3
        #boxes = [x1, y1, x2, y2] 너비 계산 -> 여러 개의 박스에 대해. 여러 개는 유지하고 값들을 가져다 쓰기

        #faster rcnn이 요청한대로.
        target = {
            'boxes':boxes,
            'labels':cls,
            'image_id': torch.tensor([index]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(len(cls),dtype=torch.uint8)        #개별 박스마다 개별 객체인지(0)(박스들이 서로 떨어진 형태), 뭉쳐있는지?
        }

        image = self.transform(image)       #이미지 형 변환(numpy -> tensor)

        return image, target

    #label_dir : 모든 라벨이 있는 폴더
    def extract_label_data(self, label_dir):
        # f -> label_dir 아래에 있는 1개의 파일 이름 ('~~.json')
        for f in sorted(os.listdir(label_dir)):
            if not f.endswith('.json'):
                continue
            #경로 읽어오고 data로 정의
            file_path = os.path.join(label_dir, f)
            print(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_name = data['images'][0]['file_name']
            image_path = os.path.join(self.image_dir, file_name)
            #boxes = bounding box , labels = object의 cls
            boxes, labels = [], []

            for ann in data['annotations']:
                x, y, w, h = ann['bbox']
                x1, y1 = x, y
                x2, y2 = x+w, y+h

                if x2 <= x1 or y2 <= y1:
                    continue

                label = NUTS_CATEGORY_MAP.get(ann['category_id'])
                if label is None:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(label)

            if not boxes:
                continue

            self.samples.append({
                'img_path': image_path,
                'boxes': boxes,
                'labels' : labels
            })



#특정 image, label 폴더를 주면 일정 비율로 쪼개서 train_image, train_label / valid_image, valid_label
def copy_split_files(file_list, image_dir, label_dir, out_image_dir, out_label_dir):
    os.mkdir(out_image_dir)
    os.mkdir(out_label_dir)

    for f in file_list:
        file_name = os.path.splitext(f)[0] + '.jpg'
        src_img = os.path.join(image_dir,file_name)
        if os.path.exists(src_img):
            #shutil.copy2(src_img, os.path.join(out_image_dir, file_name))
            print(f'{src_img}를 {os.path.join(out_image_dir, file_name)}로')    
        src_lab = os.path.join(label_dir, f)
        if os.path.exists(src_lab):
            #shutil.copy2(src_lab, os.path.join(out_label_dir, f))
            print(f'{src_lab}를 {os.path.join(out_label_dir, f)}로')            

#train, valid 쪼갠 뒤 각 폴더에 맞게 copy_split_files 수행
def split_json_files(label_dir, ratio=0.8, seed = 42):
    file_list = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])    #파일 리스트 = json으로 끝나는 f -> label_dir 내의 리스트 조회 및 정렬
    random.seed(seed)                           #seed 기준 랜덤(특정한 방식으로), 파일 리스트 랜덤으로 섞기
    random.shuffle(file_list)

    split = int(len(file_list) * ratio)         #스플릿 = 파일 리스트의 길이에 ratio 곱한 뒤 정수화
    return file_list[:split], file_list[split:] #파일리스트 돌려줌(~80, 0~20)


def get_nuts_dataloader(image_dir, label_dir):
    '''
    Nuts 이미지와 라벨을 바탕으로. train_image, train_label, valid_image, valid_label 추출함.
    앞서 정의한 copy_split_files, split_json_files를 이용해 만듦.
    '''
    os.mkdir('C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\train')
    os.mkdir('C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\valid')

    train_image = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\train\image'
    train_label = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\train\label'
    valid_image = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\valid\image'
    valid_label = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\valid\image'

    train_files, valid_files = split_json_files(label_dir, ratio = 0.8, seed=42)

    copy_split_files(file_list=train_files, image_dir  = image_dir, label_dir = label_dir, out_image_dir = train_image, out_label_dir=train_label)
    copy_split_files(file_list=valid_files, image_dir = image_dir, label_dir = label_dir, out_image_dir = valid_image, out_label_dir=valid_files)

    train_ds = NutDataset(train_image, train_label)
    valid_ds = NutDataset(valid_image, valid_label) 

    train_loader = DataLoader(train_ds, 
                              batch_size = 2,
                              shuffle = True,
                              num_workers = 0,
                              collate_fn = collate_fn)
    valid_loader = DataLoader(valid_ds, 
                              batch_size = 2,
                              shuffle = True,
                              num_workers = 0,
                              collate_fn = collate_fn)

    return train_loader, valid_loader

#배치별로 데이터 묶음 -? 분류에서는 없어도 되지만, 세그멘테이션은 라벨이 여러 개라서 필요.
#특히 object dectection에서 하나의 이미지에 여러 개의 객체가 있을 때 묶어줌. 
#pytorch model zoo의 faster rcnn 쓸 때는 필수임.
def collate_fn(batch):
    return tuple(zip(*batch))


# if __name__ == '__main__':
#     image_dir = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\images'
#     label_dir = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\labels'

#     out_image_dir = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\sample_images'
#     out_label_dir = r'C:\Users\user\Desktop\Git\sesac_localDL260810\Data\NutsDataset\sampel_labels'

#     file_list = sorted([f for f in os.listdir(label_dir) if f.endswith('.json')])
#     copy_split_files(file_list, image_dir, label_dir, out_image_dir, out_label_dir)

#    trans = None

    # nuts = NutDataset(image_dir=image_dir, label_dir=label_dir, transforms=trans)
    # print(nuts.samples)

