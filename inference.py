import os
import sys
import time
import argparse
import warnings
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.d2dewarp_model import D2DewarpModel

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return grad

def load_checkpoint(model, model_path, dev):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=dev, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=dev)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if name in model_dict:
            new_state_dict[name] = v
        elif ('module.' + name) in model_dict:
            new_state_dict['module.' + name] = v
        else:
            new_state_dict[k] = v

    matched = {k: v for k, v in new_state_dict.items() if k in model_dict}
    print(f"Loading weights from {model_path}: matched {len(matched)} / {len(model_dict)} keys")
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return model

def process_folder(model, input_dir, output_dir, benchmark_name='', resume=False, input_size=448):
    os.makedirs(output_dir, exist_ok=True)
    img_list = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    if not img_list:
        print(f"No images found in {input_dir}")
        return

    tag = benchmark_name if benchmark_name else os.path.basename(output_dir)
    t0 = time.time()

    pbar = tqdm(img_list, desc=f"{tag:<20}", unit="img", ncols=95)
    for img_path in pbar:
        name = os.path.splitext(img_path)[0]
        out_file = os.path.join(output_dir, name + '_geo.png')

        if resume and os.path.exists(out_file) and os.path.getsize(out_file) > 1024:
            continue

        full_img_path = os.path.join(input_dir, img_path)
        img = np.array(Image.open(full_img_path).convert("RGB"))
        img_h, img_w, _ = img.shape
        input_img = cv2.resize(img, (input_size, input_size)) / 255.

        edge = gradient(img)
        edge = edge[:, :, np.newaxis]
        edge = cv2.resize(edge.astype(np.uint8), (input_size, input_size)) / 255.

        with torch.no_grad():
            input_ = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(device)
            edge_ = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to(device)
            x = torch.cat((input_.float(), edge_.float()), dim=1)

            pred_h_lst, pred_v_lst, pred_bm = model(x)
            bm = (2. * (pred_bm / float(input_size)) - 1) * 1.004

            bm = bm.cpu()
            bm0 = cv2.resize(bm[0, 0].numpy(), (img_w, img_h))
            bm1 = cv2.resize(bm[0, 1].numpy(), (img_w, img_h))
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0).to(device)

            im_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device)
            out = F.grid_sample(im_tensor, lbl, align_corners=True)
            img_geo = ((out[0] * 255).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            cv2.imwrite(out_file, img_geo[:, :, ::-1])

        if device.type == 'cuda':
            vram_mb = torch.cuda.memory_allocated() / (1024**2)
            pbar.set_postfix({'VRAM': f"{vram_mb:.0f}MB"})

    elapsed = time.time() - t0
    fps = len(img_list) / max(elapsed, 1e-5)
    print(f"✅ Hoàn tất [{tag}]: {len(img_list)} ảnh trong {elapsed:.1f}s ({fps:.1f} img/s)\n")

def main():
    parser = argparse.ArgumentParser(description="D2Dewarp Inference & Benchmark Evaluation")
    parser.add_argument('--distorted_path', '--img_path', default='./distorted/', help='Path to distorted images')
    parser.add_argument('--gsave_path', '--save_path', default='./rectified/', help='Path to save dewarped images')
    parser.add_argument('--model_path', default='./model_pretrained/d2dewarp.pth', help='Path to pretrained weights (.pt or .pth)')
    parser.add_argument('--base_dir', default=None, help='Base directory of Document_Dewarp_Benchmark')
    parser.add_argument('--benchmarks', nargs='+', default=None, help='List of benchmarks or all')
    parser.add_argument('--resume', action='store_true', default=False, help='Skip already dewarped valid files')
    parser.add_argument('--input_size', type=int, default=448, help='Model input size (default: 448)')
    parser.add_argument('--d_model', type=int, default=448, help='Model dimension (default: 448)')
    parser.add_argument('--in_chans', type=int, default=4, help='Input channels (default: 4)')
    parser.add_argument('--hv_out_chans', type=int, default=1, help='HV channels (default: 1)')

    opt = parser.parse_args()

    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = D2DewarpModel(
        img_size=opt.input_size,
        in_chans=opt.in_chans,
        hv_out_chans=opt.hv_out_chans,
        d_model=opt.d_model
    ).to(device)

    load_checkpoint(model, opt.model_path, device)
    model.eval()

    if opt.base_dir:
        all_bm = ['docunet_paper_subset', 'docunet_130', 'dir300', 'docreal']
        if not opt.benchmarks or 'all' in opt.benchmarks:
            target_bms = all_bm
        else:
            target_bms = opt.benchmarks

        print(f"\n🚀 Bắt đầu nắn phẳng {len(target_bms)} tập benchmark: {target_bms} (Resume: {opt.resume})")
        for bm in target_bms:
            in_dir = os.path.join(opt.base_dir, 'benchmarks', bm, 'crop')
            if not os.path.exists(in_dir):
                in_dir = os.path.join(opt.base_dir, 'data_benchmark', 'crop', 'crop')
            out_dir = os.path.join(opt.base_dir, 'predictions', 'D2Dewarp', bm)
            process_folder(model, in_dir, out_dir, benchmark_name=bm, resume=opt.resume, input_size=opt.input_size)
    else:
        process_folder(model, opt.distorted_path, opt.gsave_path, resume=opt.resume, input_size=opt.input_size)

if __name__ == '__main__':
    main()
