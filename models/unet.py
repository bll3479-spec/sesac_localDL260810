import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

#기본 블록

#down 방향 블록

#up 방향 블록

#최종 Unet

#train
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    #tqdm -> image, mask 데이터 쌍 입력
    for imgs, masks in tqdm(loader, desc='Train'):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)          # (B, C, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
#eval -> 평가 지표
def evaluate(model, Loader, criterion, device, num_classes = 16):
    model.eval()
    total_loss = 0.0
    iou_sum = torch.zeros(num_classes)
    iou_count = torch.zeros(num_classes)

    with torch.no_grad():
        for image, mask in tqdm(Loader, desc = 'Valid'):
            image = image.to(device)
            mask = mask.to(device)

            outputs = model(image)
            loss = criterion(outputs, mask)
            total_loss += loss. item()

            #classification 예측 -> 어떤 클래스?
            #object detection 예측 -> Bbox가 어디 있고, bbox가 무엇인지?
            #segmentation 예측 -> 이 덩어리(픽셀)이 무엇인지(픽셀의 덩어리 예측)
            preds = outputs.argmax(dim=1)   #preds: 모델이 훈련 결과를 바탕으로 얻어낸 예측 값

            for cls in range(num_classes):
                pred_c = (preds == cls) #내가 예측한 픽셀
                true_c = (mask == cls)  #실제 픽셀

                #교집합 구하기
                inter = (pred_c & true_c).sum().item()  #픽셀의 겹친 부분 세기 (sum, item)
                #합집합 구하기
                union = (pred_c | true_c).sum().item()  #실제 픽셀이거나, 예측한 픽셀의 개수

                #겹침/전체
                if union > 0:
                    iou_sum[cls] += inter/union
                    iou_count[cls] += 1