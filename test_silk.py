from silk.icra25.featureness import *
import torch
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import silk.icra25.frame_score as fscore

def mask_image(img, mask, intensity=0):
    # Pad mask with false values to make it same size as image
    mask = np.pad(mask, ((0, img.shape[0] - mask.shape[0]), (0, img.shape[1] - mask.shape[1])), mode='constant', constant_values=False)
    img[mask==False] = intensity
    return img


model_dnn_path = "/home/christoa/Developer/pixer/pixer_v2/silk_data/dnn.ckpt"
model_uh_path="/home/christoa/Developer/pixer/pixer_v2/silk_data/uh_mc100.ckpt"
model_uh = SiLKDNNUncertainty(model_dnn_path, model_uh_path)

# save_path = "./tmp/"+str(prob_thresh)+"_"+str(uncer_thresh)+"/"
frame = 000


# Canals frame
img_left_path = "/home/christoa/Developer/pixer/pixer_v2/silk_data/frame.png"

img_left = load_images(img_left_path)


img_left_cv = cv2.imread(img_left_path)
img_left_og = img_left_cv.copy()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_uh = model_uh.to(DEVICE)
prob_mean_1, prob_var_1, _ = model_uh(img_left)
# prob_mean_2, prob_var_2, _ = model_uh(img_left)


prob_thresh = 0.0 # Everything above this is considered as a feature
uncer_thresh = 0.1 # Everything below this is considered as a feature

score_image_0 = fscore.featureness_image(prob_mean_1, prob_var_1, prob_thresh=prob_thresh, uncer_thresh=uncer_thresh).detach().cpu().numpy()

score_image_0 = score_image_0.squeeze()

zero_rows = max(img_left.shape[2] - score_image_0.shape[0], 0)//2
zero_cols = max(img_left.shape[3] - score_image_0.shape[1], 0)//2
# score_image_0 = np.pad(score_image_0, ((0,zero_rows), (0,zero_cols)), mode='constant', constant_values=0)
score_image_0 = np.pad(score_image_0, ((zero_rows,0), (zero_cols,0)), mode='constant', constant_values=0)

print (score_image_0.shape)
print (img_left_cv.shape)


img_left_cv = cv2.cvtColor(img_left_cv, cv2.COLOR_BGR2RGB)
img1_masked = mask_image(img_left_cv,score_image_0, intensity=0)
# Swap R and B channels in matplotlib


print ("Saving image: ", frame)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[2].imshow(img1_masked)
axes[2].set_title('Masked Image')
axes[0].imshow(img_left_og)
axes[0].set_title('Original Image')
axes[1].imshow(score_image_0)
axes[1].set_title('Score Image')
plt.show()


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



