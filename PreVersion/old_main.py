#모델 zoo : https://docs.pytorch.org/serve/model_zoo.html
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision.models import vgg11, VGG11_Weights

#내가 만든 파일, 폴더에서 함수 불러오기
from utils.DataLoader import get_dataloader
from models.vgg import get_vgg_model
from train import train_one_epoch
from eval import evaluate
from utils.graph import draw_plot
from models.resnet_pre import get_resnet_model

from tqdm import tqdm

#import train , train 파일에 있는 함수를 가져오기 : train.train_one_epoch로 주소 찍기
#아니면 from train import *로 전역 정의

#전처리
#데이터셋로드
#훈련함수
#검증함수
#필요시 시각화


def run_epoch(model, history, EPOCHS=10):
    #GPU 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #에포크마다 train_one_epoch랑 eval 함수를 돌리면 됨, 이후로 기록&save
    #오차 함수, 최적화 함수, 히스토리 딕셔너리 추가(wandb처럼), 에포크 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    best_acc = 0.0

    progress_bar = tqdm(range(1,EPOCHS+1))

    for e in progress_bar:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion = criterion, optimizer=optimizer, device = device)
            valid_loss, valid_acc = evaluate(model, valid_loader, criterion = criterion, optimizer=optimizer, device = device)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['valid_loss'].append(valid_loss)
            history['valid_acc'].append(valid_acc)
            progress_bar.set_postfix({'train_loss':f'{train_loss:.2f}',
                                      'train_acc':f'{train_acc:.2f}',
                                      'valid_loss':f'{valid_loss:.2f}',
                                      'valid_acc': f'{valid_acc:.2f}'
                                      })

            if valid_acc > best_acc: 
                best_acc = valid_acc
                save_path = f'./{e+1} epoch_{valid_acc:.2f}.pth'
                torch.save(model.state_dict(), save_path)
                tqdm.write(f'최고 기록 갱신 -> {e+1}에포크에서 {valid_acc:.2f}')
    return history


if __name__ == '__main__':

    #데이터를 로딩하는 함수
    train_loader, valid_loader = get_dataloader()

    #vgg11 모델과 가중치를 로딩하는 함수
    model = get_resnet_model()
    # print(model)
    history = {'train_loss':[], 'train_acc':[], 'valid_loss':[], 'valid_acc':[]}
    #model_history: 훈련이 끝났을 때의 총 history 반환
    model_history = run_epoch(model, history, EPOCHS=5)

    draw_plot(model_history, save_path=r'./result.jpg')

    
    #return history, best_acc

    #images, labels = next(iter(train_loader))
    #print(labels.size())

