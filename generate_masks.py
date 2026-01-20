from silk.icra25.featureness import *
import torch
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import silk.icra25.frame_score as fscore
import os
from tqdm import tqdm

def mask_image(img, mask, intensity=0):
    # Pad mask with false values to make it same size as image
    mask = np.pad(mask, ((0, img.shape[0] - mask.shape[0]), (0, img.shape[1] - mask.shape[1])), mode='constant', constant_values=False)
    img[mask==False] = intensity
    return img

img_dir = '/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/image_0/'
mean = '/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/mean/'
var = '/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/var/'

if not os.path.exists(mean):
    os.makedirs(mean)
if not os.path.exists(var):
    os.makedirs(var)

model_dnn_path = "/home/christoa/Developer/pixer/pixer_v2/silk_data/dnn.ckpt"
model_uh_path="/home/christoa/Developer/pixer/pixer_v2/silk_data/uh_mc100.ckpt"
model_uh = SiLKDNNUncertainty(model_dnn_path, model_uh_path)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_uh = model_uh.to(DEVICE).eval()

imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])

for frame_loc in tqdm(imgs[:1600]):

    img_left = load_images(frame_loc)
    if img_left.ndim != 4 or img_left.shape[0] != 1 or img_left.shape[1] != 1:
        raise ValueError(
            f"Expected input shape (1,1,H,W), got {tuple(img_left.shape)}"
        )

    prob_mean_1, prob_var_1, _ = model_uh(img_left)
    stem = os.path.splitext(os.path.basename(frame_loc))[0]
    np.save(os.path.join(mean, f"{stem}_mean.npy"),
                    prob_mean_1.detach().cpu().numpy().astype(np.float32))
    np.save(os.path.join(var,  f"{stem}_var.npy"),
            prob_var_1.detach().cpu().numpy().astype(np.float32))
