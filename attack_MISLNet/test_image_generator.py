import image_generator
import cv2
import numpy as np
im_path = '/media/nas2/Shengbang/Columbia/4cam_splc/canong3_canonxt_sub_01.tif'
patch_size = 128
batch_size = 3
generator = image_generator.ImagePatchGenerator(im_path, patch_size, batch_size)

def test(generator):
    input_im = cv2.imread(im_path)
    print('input image shape:', input_im.shape)
    iteration = generator.init_generator()
    #for p in generator.patch_list:
    #    print(p.shape)
    print('iter:', iteration)
    batch_list = []
    for ii in range(iteration):
        batch = generator.get_next()
        batch_list.append(batch)
    output_batch = np.concatenate(batch_list, 0)
    output = generator.construct_attack_result(output_batch)
    print(output.shape)
    print(np.mean(input_im - output))

test(generator)
test(generator)
