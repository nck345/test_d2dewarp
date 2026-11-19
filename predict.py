"""
Official Code Implementation of:
"D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping"
"""

import argparse
import time
import cv2
import glob
import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from networks.d2dewarp_model import D2DewarpModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return grad

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=448, help='image size')
    parser.add_argument('--model_path', default='./model_weight.pt', help='model path')
    parser.add_argument('--img_path', default='./distorted/', help='image path or folder')
    parser.add_argument('--save_path', default='./rectified/', help='save path')
    parser.add_argument('--hv_out_chans', type=int, default=1, help='h line and v line output channels')
    parser.add_argument('--d_model', type=int, default=448, help='last layer dim in UNet and all layers in attention')
    parser.add_argument('--in_chans', type=int, default=4, help='input channels')
    return parser.parse_args()

def predict(img_path, save_path, filename, recti_model, input_size=448):
    assert os.path.exists(img_path), f'Incorrect Image Path: {img_path}'
    os.makedirs(save_path, exist_ok=True)

    dewarp_path = os.path.join(save_path, 'dewarp')
    pred_h_path = os.path.join(save_path, 'pred_h')
    pred_v_path = os.path.join(save_path, 'pred_v')
    os.makedirs(dewarp_path, exist_ok=True)
    os.makedirs(pred_h_path, exist_ok=True)
    os.makedirs(pred_v_path, exist_ok=True)

    img = np.array(Image.open(img_path).convert("RGB"))
    img_h, img_w, _ = img.shape
    input_img = cv2.resize(img, (input_size, input_size)) / 255.

    edge = gradient(img)
    edge = edge[:, :, np.newaxis]
    edge = cv2.resize(edge.astype(np.uint8), (input_size, input_size)) / 255.

    recti_model.eval()

    with torch.no_grad():
        input_ = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device)
        edge_ = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to(device)
        start = time.time()
        pred_h_lst, pred_v_lst, pred_bm = recti_model(torch.cat((input_.float(), edge_.float()), dim=1))
        bm = (2. * (pred_bm / float(input_size)) - 1) * 1.004
        ps_time = time.time() - start

    bm = bm.cpu()
    bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))
    bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))

    bm0 = cv2.blur(bm0, (3, 3))
    bm1 = cv2.blur(bm1, (3, 3))
    lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).to(device)

    im_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device)
    out = F.grid_sample(im_tensor, lbl, align_corners=True)
    img_geo = ((out[0] * 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
    base_name = os.path.basename(filename)
    cv2.imwrite(os.path.join(dewarp_path, base_name), img_geo[:, :, ::-1])

    pred_h = pred_h_lst[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_h = (pred_h > 0.2)
    pred_h = (pred_h * 255.).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_h_path, base_name), cv2.resize(pred_h, (img_w, img_h)))

    pred_v = pred_v_lst[-1].squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_v = (pred_v > 0.2)
    pred_v = (pred_v * 255.).astype(np.uint8)
    cv2.imwrite(os.path.join(pred_v_path, base_name), cv2.resize(pred_v, (img_w, img_h)))

    return ps_time

def main():
    args = get_args()
    print(f"Device: {device}")

    recti_model = D2DewarpModel(
        img_size=args.input_size,
        in_chans=args.in_chans,
        hv_out_chans=args.hv_out_chans,
        d_model=args.d_model
    ).to(device)

    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    model_dict = recti_model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if name in model_dict:
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    model_dict.update({k: v for k, v in new_state_dict.items() if k in model_dict})
    recti_model.load_state_dict(model_dict)
    print('Model loaded successfully')

    if os.path.isdir(args.img_path):
        files = glob.glob(os.path.join(args.img_path, "*"))
    else:
        files = [args.img_path]

    total_time = 0.0
    img_num = 0

    for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        print("Processing:", file)
        base = os.path.splitext(os.path.basename(file))[0]
        out_name = base + ".png"
        t = predict(file, args.save_path, out_name, recti_model, input_size=args.input_size)
        total_time += t
        img_num += 1

    if img_num > 0:
        print('FPS: %.1f' % (img_num / max(total_time, 1e-5)))

if __name__ == '__main__':
    main()
