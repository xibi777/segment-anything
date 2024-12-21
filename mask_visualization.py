# import cv2
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# sam = sam_model_registry["vit_h"](checkpoint="/media/data/ziqin/pretrained/sam_vit_h_4b8939.pth")
# mask_generator = SamAutomaticMaskGenerator(sam)
# image = cv2.imread('/media/data/ziqin/data/VOCdevkit/VOC2012/JPEGImages/2011_006671.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# masks = mask_generator.generate(image)
# print('ll')

# # mask visualization
# import cv2
# import numpy as np
# import os

# PALETTE = [[240,128,128], [0, 192, 64], [0, 64, 96], [128, 192, 192],
#                [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
#                [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
#                [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
#                [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
#                [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
#                [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
#                [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
#                [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
#                [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
#                [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
#                [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
#                [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
#                [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
#                [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
#                [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
#                [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
#                [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
#                [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
#                [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
#                [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
#                [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
#                [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
#                [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
#                [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
#                [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
#                [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
#                [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
#                [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
#                [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
#                [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
#                [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
#                [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
#                [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
#                [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
#                [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
#                [0, 192, 32], [238, 209, 156], [64, 0, 64], [128, 192, 160],
#                [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
#                [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
#                [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
#                [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
#                [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
#                [64, 192, 96], [64, 160, 64], [64, 64, 0]]


# file_path = '/media/data/ziqin/output_sam/SAM_test_5'
# for i, file in enumerate(os.listdir(file_path)):
#     img = cv2.imread(os.path.join('/media/data/ziqin/data/VOCdevkit/VOC2012/JPEGImages', str(file) + '.jpg'))
#     mask_names = os.listdir(os.path.join(file_path,file))
#     mask_names = [f for f in mask_names if 'png' in f]
#     mask_full = np.zeros_like(img)
#     for mask_name in mask_names:
#         mask = cv2.imread(os.path.join(file_path,file, mask_name))
#         mask = mask / 255
#         # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#         color = PALETTE[i % len(PALETTE)]
#         mask[:, :, 0] =  mask[:, :, 0] * color[0]
#         mask[:, :, 1] =  mask[:, :, 1] * color[1]
#         mask[:, :, 2] =  mask[:, :, 2] * color[2]
#         mask = mask.astype(img.dtype)
#         alpha = 0.3
#         beta = 1 - alpha
#         blended_image = cv2.addWeighted(img, alpha, mask, beta, 0)
#         cv2.imwrite(os.path.join(file_path, file, 'mask' + mask_name), blended_image)
        
#         mask_full = mask_full + mask
#     # mask_full = mask_full.astype(img.dtype)
#     # alpha = 0.2
#     # beta = 1 - alpha
#     mask_full_image = cv2.addWeighted(img, alpha, mask_full, beta, 0)
#     cv2.imwrite(os.path.join(file_path, file, 'mask_full.jpg'), mask_full_image)
#     print(str(file) + ' finished')

# mask process
import cv2
import numpy as np
import os

# PALETTE = [[240,128,128], [0, 192, 64], [0, 64, 96], [128, 192, 192],
#                [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
#                [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
#                [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
#                [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
#                [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
#                [64, 128, 32], [0, 160, 0], [192, 128, 160],
#                [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
#                [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
#                [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
#                [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
#                [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
#                [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
#                [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
#                [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
#                [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
#                [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
#                [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
#                [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
#                [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
#                [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
#                [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
#                [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
#                [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
#                [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
#                [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
#                [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
#                [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
#                [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
#                [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
#                [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
#                [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
#                [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
#                [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
#                [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
#                [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
#                [0, 192, 32], [238, 209, 156], [64, 0, 64], [128, 192, 160],
#                [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
#                [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
#                [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
#                [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
#                [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
#                [64, 192, 96], [64, 160, 64], [64, 64, 0]]

file_path = '/media/data/ziqin/output_sam/SAM_masks'
out_path = '/media/data/ziqin/output_sam/SAM_masks_process_0'
iou_threshold = 0.9
#zyq
for i, file in enumerate(os.listdir(file_path)):
    # color = PALETTE[i % len(PALETTE)]
    # img = cv2.imread(os.path.join('/media/data/ziqin/data/VOCdevkit/VOC2012/JPEGImages', str(file) + '.jpg'))
    # print(img.dtype)
    mask_names = os.listdir(os.path.join(file_path,file))
    mask_names = [f for f in mask_names if 'png' in f and 'mask' not in f]
    mask_full = []
    
    for mask_name in mask_names:
        mask = cv2.imread(os.path.join(file_path,file, mask_name))
        mask_bool = mask.astype(bool)
        mask_full.append(mask_bool)
    mask_full_check = mask_full.copy()
    for index, mask_i in enumerate(mask_full):
        mask_full_ex = [arr for arr in mask_full_check if not np.array_equal(arr, mask_i)]
        for mask_j in mask_full_ex:
            intersection = np.logical_and(mask_i[:, :, 0], mask_j[:, :, 0])
            check_iou = np.sum(intersection) / np.sum(mask_i[:, :, 0])
            # print(check_iou)
            if check_iou > iou_threshold:
                mask_full_check = [arr for arr in mask_full_check if not np.array_equal(arr, mask_i)]
    for n, mask_retain in enumerate(mask_full_check):
        mask_retain = mask_retain.astype('uint8')
        mask_retain = mask_retain * 255
        img_path = os.path.join(out_path, file)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        cv2.imwrite(os.path.join(out_path, file, str(n) + '.png'), mask_retain)
    print(str(file) + ' finished')
    
    # mask visualization 
    # mask_full_new = np.zeros_like(img)
    # for n, mask_retain in enumerate(mask_full_check):
    #     mask_retain = mask_retain.astype(img.dtype)
    #     mask_retain[:, :, 0] =  mask_retain[:, :, 0] * color[0]
    #     mask_retain[:, :, 1] =  mask_retain[:, :, 1] * color[1]
    #     mask_retain[:, :, 2] =  mask_retain[:, :, 2] * color[2]
    #     mask_retain = mask_retain.astype(img.dtype)
    #     alpha = 0.3
    #     beta = 1 - alpha
    #     blended_image = cv2.addWeighted(img, alpha, mask_retain, beta, 0)
    #     img_path = os.path.join(out_path, file)
    #     if not os.path.exists(img_path):
    #         os.makedirs(img_path)
    #     cv2.imwrite(os.path.join(out_path, file, 'mask'+ str(n) + '.jpg'), blended_image)
    #     mask_full_new = mask_full_new + mask_retain
    # mask_full_image = cv2.addWeighted(img, alpha, mask_full_new, beta, 0)
    # cv2.imwrite(os.path.join(out_path, file, 'mask_full_0.jpg'), mask_full_image)
    # print(str(file) + ' finished')
                
            
            
        
