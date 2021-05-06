import numpy as np
import glob
import cv2
import tqdm
from authentic_patch_extractor import AuthPatchExtractor
import random

columbia_im_path = '/media/nas2/Shengbang/Columbia/4cam_splc/'
columbia_ma_path = '/media/nas2/Shengbang/Columbia/4cam_splc/edgemask/'

columbia_im = glob.glob(columbia_im_path + '*.tif')
random.shuffle(columbia_im)

im_path = columbia_im[0]
name = im_path.split('/')[-1].split('.')[0]
print(name)
ma_list = glob.glob(columbia_ma_path + name + '*')
ma_path = ma_list[0]

print(im_path, ma_path)
extractor = AuthPatchExtractor(im_path, ma_path)

auth_patches, splc_patches = extractor.dynamic_extract_patches(patch_size=128)
'''
image = cv2.imread(im_path)
for start_y in range(0, image.shape[0], 128):
    cv2.line(image, (0, start_y), (image.shape[1], start_y), (0,0,0), 4)
for start_x in range(0, image.shape[1], 128):
    cv2.line(image, (start_x, 0), (start_x, image.shape[0]), (0,0,0), 4)
cv2.imwrite('test_org.png', image)
cv2.imwrite('show_mask.png', extractor.show_mask())
for ii, auth_p in enumerate(auth_patches):
    cv2.imwrite('test_auth_' + str(ii) + '.png', auth_p)
for ii, splc_p in enumerate(splc_patches):
    cv2.imwrite('test_splc_' + str(ii) + '.png', splc_p)
'''
print(len(auth_patches), len(splc_patches))
