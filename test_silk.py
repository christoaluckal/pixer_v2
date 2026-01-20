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
model_uh = model_uh.to(DEVICE)

imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])

for frame_loc in tqdm(imgs):

    img_left = load_images(frame_loc)


    prob_mean_1, prob_var_1, _ = model_uh(img_left)

    prob_mean_1 = prob_mean_1.squeeze().detach().cpu().numpy()
    prob_var_1 = prob_var_1.squeeze().detach().cpu().numpy()

    np.save(os.path.join(mean, os.path.basename(frame_loc).replace('.png', '_mean.npy')), prob_mean_1)
    np.save(os.path.join(var, os.path.basename(frame_loc).replace('.png', '_var.npy')), prob_var_1)


# prob_thresh = np.linspace(0,1,20)
# uncer_thresh = np.linspace(0,1,20)
# image_area = img_left.shape[2] * img_left.shape[3]
# cutoff_area = 0.3 * image_area

# for pt in prob_thresh:
#     for ut in uncer_thresh:
#         score_image_0 = fscore.featureness_image(prob_mean_1, prob_var_1, prob_thresh=pt, uncer_thresh=ut).detach().cpu().numpy()
#         score_image_0 = score_image_0.squeeze()
        
#         masked_img = mask_image(img_left_cv.copy(), score_image_0, intensity=0)
#         masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
#         non_zero_pixels = np.sum(masked_img_gray > 0)
#         if non_zero_pixels < cutoff_area:
#             print(f'Non-zero pixels: {non_zero_pixels}, area: {cutoff_area}')
#             # print (f"Skipping pt: {pt}, ut: {ut}, area: {non_zero_pixels}")
#             continue
#         cv2.imwrite(f"silk_data/img_masked_{pt}_{ut}.png", masked_img)



