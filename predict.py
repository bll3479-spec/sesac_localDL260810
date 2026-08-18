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

    image_target = Image.open(image_path).convert('RGB')
    inp = transforms(image_target).unsqeeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(inp)

    pred_mask = output.argmax(dim=1).squeeze(0).cpu().memory()
    display_image = np.array(image_target.resize(image_size, image_size))
    return pred_mask, display_image

if __name__ == '__main__':
    args = parse_args()
    if args.option == '안녕':
        print('안녕')