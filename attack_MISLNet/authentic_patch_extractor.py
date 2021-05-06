import numpy as np
import cv2
import glob
import tqdm
import math

class AuthPatchExtractor:
    def __init__(self, im_path, ma_path, ma_type='columbia'):
        self.im = cv2.imread(im_path)
        self.ma = cv2.imread(ma_path)
        if(ma_type == 'columbia'):
            self.ma_columbia()
        elif(ma_type == 'carvalho'):
            self.ma_carvalho()
        else:
            raise ValueError('ma_type should be either columbia or carvalho')
        print(self.ma.shape)
        self.auth_list = []
        self.splc_list = []

    def show_mask(self):
        image = self.im.copy()
        for x in range(self.im.shape[0]):
            for y in range(self.im.shape[1]):
                image[x, y, 2] = 0
                image[x, y, 1] = int(self.im[x, y, 1] * self.ma[x, y])
                image[x, y, 0] = int(self.im[x, y, 0] * (1 - self.ma[x, y]))
        return image

    def ma_columbia(self):
        self.ma = self.ma[..., 1]
        self.ma = self.ma > 128
        if(np.mean(self.ma) > 0.5):
            self.ma = 1 - self.ma

    def ma_carvalho(self):
        self.ma = np.mean(self.ma, axis=-1)
        self.ma = self.ma > 128
        if(np.mean(self.ma) > 0.5):
            self.ma = 1 - self.ma

    def dynamic_extract_patches(self, patch_size=128, step_size=10, tolerance=1e-6):
        # search for the best split.
        auth_list = []
        splc_list = []
        for start_x in list(range(0, patch_size, step_size)):
            for start_y in list(range(0, patch_size, step_size)):
                auth_l, splc_l = self.extract_patches(start_x=start_x, start_y=start_y, patch_size=patch_size, tolerance=tolerance)
                if(len(auth_l) > len(auth_list)):
                    auth_list = auth_l
                if(len(splc_l) > len(splc_list)):
                    splc_list = splc_l
        return auth_list, splc_list

    def extract_patches(self, start_x=0, start_y=0,  patch_size=128, tolerance=1e-6):
        # tolerance means the percentage of mixing auth/splc parts.
        
        image = self.im[start_x:, start_y:]
        mask = self.ma[start_x:, start_y:]
     
        size_x = image.shape[0] - start_x
        size_y = image.shape[1] - start_y
        
        N_x = math.ceil(size_x / patch_size)
        N_y = math.ceil(size_y / patch_size)

        auth_list = []
        splc_list = []

        for ii in range(N_x):
            for jj in range(N_y):
                st_x = ii * patch_size
                st_y = jj * patch_size
                ed_x = (ii + 1) * patch_size
                ed_y = (jj + 1) * patch_size
                if(ed_x > size_x):
                    ed_x = size_x
                    st_x = ed_x - patch_size
                if(ed_y > size_y):
                    ed_y = size_y
                    st_y = ed_y - patch_size
                this_ma = mask[st_x : ed_x, st_y : ed_y]
                
                mean = np.mean(this_ma)
                if(mean <= tolerance):
                    # authentic
                    auth_list.append(image[st_x : ed_x, st_y : ed_y])
                elif(mean >= 1 - tolerance):
                    # spliced
                    splc_list.append(image[st_x : ed_x, st_y : ed_y])
                else:
                    continue


        self.auth_list = auth_list
        self.splc_list = splc_list
        return auth_list, splc_list

        
    

