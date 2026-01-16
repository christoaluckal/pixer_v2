# pyslam/semantics/silk_mask.py
import numpy as np
import cv2
import torch
import silk.icra25.frame_score as fscore
from silk.icra25.featureness import SiLKDNNUncertainty  # adjust to what you actually use
import matplotlib.pyplot as plt

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
    
if __name__ == "__main__":
    # Example usage
    dnn_ckpt = "/home/christoa/Developer/pixer/pixer_v2/silk_data/dnn.ckpt"
    uh_ckpt = "/home/christoa/Developer/pixer/pixer_v2/silk_data/uh_mc100.ckpt"
    masker = SilkMaskGenerator(dnn_ckpt, uh_ckpt, prob_thresh=0.5, uncer_thresh=0.1)

    img = cv2.imread("/home/christoa/Developer/pixer/pixer_v2/silk_data/frame.png")
    mask = masker(img)

    # cv2.imwrite("path/to/mask.png", mask * 255)  # Save mask as binary image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Input Image")
    axs[0].axis("off")
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Generated Mask")
    axs[1].axis("off")
    plt.show()