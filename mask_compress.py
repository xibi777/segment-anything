import cv2
import numpy as np
import os

file_path = '/media/data/ziqin/output_sam/SAM_masks_process_1'
out_path = '/media/data/ziqin/output_sam/SAM_masks_process_2'
for i, file in enumerate(os.listdir(file_path)):
    mask_names = os.listdir(os.path.join(file_path,file))
    mask_names = [f for f in mask_names if 'png' in f]
    mask_full = []
    
    for mask_name in mask_names:
        mask = cv2.imread(os.path.join(file_path,file, mask_name))
        mask_full.append(mask[:, :, 0])
    
    mask_one = np.ones_like(mask[:, :, 0]) * 255     
    for j, mask_j in enumerate(mask_full):
        mask_one = np.where(mask_j == 255, j, mask_one)

    img_path = os.path.join(out_path, file)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    cv2.imwrite(os.path.join(out_path, file, 'mask.png'), mask_one)
    print(str(file) + ' finished')