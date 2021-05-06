import numpy as np
import cv2
import math

class ImagePatchGenerator():
    def __init__(self, im_path, patch_size, batch_size):
        try:
            self.image = cv2.imread(im_path)
        except:
            raise ValueError('No image in this path')
        self.shape = self.image.shape
        self.H = self.shape[0]
        self.W = self.shape[1]
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.stride_H = 0
        self.stride_W = 0
        self.N_H = 0
        self.N_W = 0
        self.patch_list = []
        self.generator = None
        self.attack_list = []
        self.attack_result = None
        
    def get_next(self):
        return next(self.generator)

    def reset(self):
        del self.generator
        self.generator = generator()

    def get_generator(self):
        out_batch = []
        for ii, patch in enumerate(self.patch_list):
            out_batch.append(patch)
            if(len(out_batch) == self.batch_size):
                output_array = np.array(out_batch)
                out_batch.clear()
                yield output_array

    def init_generator(self, overlap=0):
        iteration = self.get_patch_list(overlap)
        self.generator = self.get_generator()
        return iteration
    
    def get_patch_list(self, overlap=None):
        # overlap: # of pixels that is overlapped
        if(overlap is None):
            self.N_H = math.ceil(self.H / self.patch_size)
            self.N_W = math.ceil(self.W / self.patch_size)
            self.stride_H = math.ceil((self.H - self.patch_size) / (self.N_H - 1))
            self.stride_W = math.ceil((self.W - self.patch_size) / (self.N_W - 1))
        else:
            self.stride_H = self.patch_size - overlap
            self.stride_W = self.stride_H
            self.N_H = math.ceil((self.H - self.patch_size) / self.stride_H) + 1
            self.N_W = math.ceil((self.W - self.patch_size) / self.stride_W) + 1

        #print('start get patches')
        for ii in range(self.N_H):
            for jj in range(self.N_W):
                start_x, end_x, start_y, end_y = self.get_position(ii, jj)
                #print('end pt:', end_x, end_y)
                self.patch_list.append(self.image[start_x : end_x, start_y : end_y])
        repeat = math.ceil(len(self.patch_list) / self.batch_size) * self.batch_size - len(self.patch_list)
        self.patch_list.extend(self.patch_list[: repeat])
        iteration = math.ceil(len(self.patch_list) / self.batch_size)
        return iteration
    
    def get_position(self, ii, jj):
        # iith row and jjth column of the patch. 
        # return start/end x/y pixel index.
        start_x = self.stride_H * ii
        start_y = self.stride_W * jj
        end_x = start_x + self.patch_size
        end_y = start_y + self.patch_size
        if(end_x > self.H):
            start_x = self.H - self.patch_size
            end_x = self.H
        if(end_y > self.W):
            start_y = self.W - self.patch_size
            end_y = self.W
        return start_x, end_x, start_y, end_y

    def construct_attack_result(self, batch):
        # batch of attack results.
        self.attack_list = batch
        self.attack_result = np.zeros_like(self.image)
        #print('start splicing')
        for ii in range(self.N_H):
            for jj in range(self.N_W):
                idx = ii * self.N_W + jj
                patch = self.attack_list[idx]
                start_x, end_x, start_y, end_y = self.get_position(ii, jj)
                #print('end pt:', end_x, end_y)
                self.attack_result[start_x : end_x, start_y : end_y] = patch
        return self.attack_result



        

    

