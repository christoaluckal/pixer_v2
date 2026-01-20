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

def stats(name, x):
    x = x.detach().cpu()
    print(name, "shape", tuple(x.shape),
          "min", float(x.min()),
          "p50", float(x.median()),
          "p90", float(x.quantile(0.90)),
          "p99", float(x.quantile(0.99)),
          "max", float(x.max()))
    
class SilkMaskGenerator:
    def __init__(self, dnn_ckpt, uh_ckpt, prob_thresh=0.0, uncer_thresh=0.1, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiLKDNNUncertainty(dnn_ckpt, uh_ckpt).to(self.device).eval()
        self.prob_thresh = prob_thresh
        self.uncer_thresh = uncer_thresh

    @torch.no_grad()
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        H, W = img_bgr.shape[:2]

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # (H,W) uint8 0..255
        x = torch.from_numpy(img_gray).to(self.device).float() / 255.0  # <-- critical
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        prob_mean, prob_var, _ = self.model(x)

        score = fscore.featureness_image(
            prob_mean, prob_var,
            prob_thresh=self.prob_thresh,
            uncer_thresh=self.uncer_thresh,
        ).detach().cpu().numpy().squeeze()

        mask = (score > 0).astype(np.uint8)

        # enforce exact size (avoid later OOB)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return mask

model_dnn_path = "/home/christoa/Developer/pixer/pixer_v2/silk_data/dnn.ckpt"
model_uh_path="/home/christoa/Developer/pixer/pixer_v2/silk_data/uh_mc100.ckpt"
model_uh = SiLKDNNUncertainty(model_dnn_path, model_uh_path)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_uh = model_uh.to(DEVICE).eval()

mask_gen = SilkMaskGenerator(
            dnn_ckpt=f"/home/christoa/Developer/pixer/pixer_v2/silk_data/dnn.ckpt",
            uh_ckpt=f"/home/christoa/Developer/pixer/pixer_v2/silk_data/uh_mc100.ckpt",
            prob_thresh=0.071,
            uncer_thresh=0.087,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

# imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
# imgs = ['/home/christoa/Developer/pixer/pixer_v2/silk_data/frame.png','/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/image_0/000030.png']

# for i, frame_loc in enumerate(imgs):

#     img_left = load_images(frame_loc)
#     prob_mean_1, prob_var_1, _ = model_uh(img_left)
#     score_image = fscore.featureness_image(prob_mean_1, prob_var_1, 0.071, 0.087)
#     # score_image = score_image.cpu().detach().numpy().squeeze().squeeze()

#     if i == 0:
#         stats("dummy mean",prob_mean_1)
#         stats("dummy unc", prob_var_1)
#         score_image = score_image.cpu().detach().numpy().squeeze().squeeze()
#         plt.imshow(score_image)
#         plt.draw()
#         plt.pause(0)
#     else:
#         stats("actual mean",prob_mean_1)
#         stats("actual unc", prob_var_1)
#         score_image = score_image.cpu().detach().numpy().squeeze().squeeze()
#         plt.imshow(score_image)
#         plt.draw()
#         plt.pause(0)


# imgs = ['/home/christoa/Developer/pixer/pixer_v2/silk_data/frame.png','/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/image_0/000030.png']
# for i, frame_loc in enumerate(imgs):

#     score_img = mask_gen(cv2.imread(frame_loc))


#     plt.imshow(score_img)
#     plt.draw()
#     plt.pause(0)



img_path = "/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/image_0/000030.png"
mean_path = "/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/mean/000030_mean.npy"
var_path  = "/home/christoa/Downloads/torrents/data_odometry_gray/dataset/sequences/00/var/000030_var.npy"

# Load image
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Load maps (keep shape consistent)
pm = torch.from_numpy(np.load(mean_path)).float()
pv = torch.from_numpy(np.load(var_path)).float()

# Score
score = fscore.featureness_image(pm, pv, 0.083, 0.243)
score = score.squeeze().detach().cpu().numpy()

# Binary mask
mask = (score > 0).astype(np.uint8)

mask = cv2.resize(mask, (img.shape[1],img.shape[0]))

img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 2) Create green overlay
green = np.zeros_like(img_bgr)
green[..., 1] = mask * 255  # green channel

# 3) Alpha blend
alpha = 0.5
overlay = cv2.addWeighted(img_bgr, 1 - alpha, green, alpha, 0)

# Plot 1x3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("Input image")
axes[0].axis("off")

axes[1].imshow(score, cmap="viridis")
axes[1].set_title("Featureness score")
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].set_title("Mask (score > 0)")
axes[2].axis("off")

plt.tight_layout()
plt.show()