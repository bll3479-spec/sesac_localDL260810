import argparse

import torch
import numpy as np
from pathlib import Path

from PIL import Image
from torchvision import transforms

from models.unet import Unet
from preprocessing.unet_preprocessing import NUM_CLASSES
from utils.visualize import draw_segmentation_mask


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', type=str, required=True, help='테스트할 이미지 경로')
    return parser.parse_args()


def predict(image_size, model, device, image_path):
    transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])

    image_target = Image.open(image_path).convert('RGB') #채널, 너비, 높이인 상태, unsqueeze로 배치 추가
    inp = transforms(image_target).unsqeeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(inp)
    #시각화용 squeeze
    pred_mask = output.argmax(dim=1).squeeze(0).cpu().memory()
    display_image = np.array(image_target.resize(image_size, image_size))
    return pred_mask, display_image



if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Unet(in_channel=3, num_classes=NUM_CLASSES, base_ch=64)
    model.load_state_dict(torch.load('./unet_nuts.pth', map_location=device))
    model.to(device=device)

    pred_mask, display_image = predict(512, model, device, './test.jpg')

    #시각화
    detected = np.unique(pred_mask)
    detected = detected[detected != 0]

    draw_segmentation(image = display_image, mask=pred_mask, alpha = 0.9, save_path = './result.jpg')

    #if args.option == '안녕':
    #    print('안녕')