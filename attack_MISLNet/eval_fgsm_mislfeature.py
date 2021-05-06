import tensorflow as tf
from os.path import join
import sys, os
sys.path.append('../')
import image_generator 
from data_pipe import TFRecordDataset
from MISLNet_Features import MISLNet_Features
from cleverhans.utils_tf import tf_model_load
from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import MSE_loss
import numpy as np
import PIL
import tqdm
import glob
import cv2

im_path = '/media/nas2/Shengbang/Columbia/4cam_splc/canong3_canonxt_sub_01.tif'
patch_size = 256
batch_size= 1
generator = image_generator.ImagePatchGenerator(im_path, patch_size, batch_size)

DB_NUM = 10
model_path = 'model-18'
#'Motorola_X_val.tfrecords'
learning_rate = 1

image_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3], name='features')

model = MISLNet_Features('MISLNet_P')
preds = model.get_logits(image_ph)

ref_patch = None
with tf.Session() as sess:
    iterations = generator.init_generator()
    ref_patch = generator.get_next()[0]
    iterations = generator.init_generator()
    print('ref_patch size:', ref_patch.shape)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    validation_iters= iterations

    tf_model_load(sess, model_path)
    references = sess.run(preds, feed_dict = {image_ph: ref_patch[np.newaxis, ...]})
    fgsm = FastGradientMethod(model, sess=sess)
    fgsm_params = {'y_target': references,
            'loss_fn': MSE_loss,
            'eps': learning_rate,
            'ord': np.inf,
            'clip_max': 255,
            'clip_min': 0}

    atk_list=[]
    for ii in tqdm.tqdm(range(iterations)):
        iix_val = generator.get_next()
        print('org shape:', iix_val.shape)
        adv, grad = fgsm.generate_np(iix_val, **fgsm_params)
        adv = adv.astype(np.uint8)
        print('adv shape:', adv.shape, grad.shape)
        # Generate adversarial examples and return them as a NumPy array.
        percent_perturbed = np.mean(np.sum((adv - iix_val) ** 2, axis=(1, 2, 3)) ** .5)
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
        avg_perturb = np.mean(np.power(adv - iix_val, 2))
        print('Avg. L2 scores:', avg_perturb)
        attacked_images = adv
        attacked_images = attacked_images.astype(np.uint8)
        atk_list.append(attacked_images)
    output = generator.construct_attack_result(atk_list)
    name = im_path.split('/')[-1]
    cv2.imwrite('./results/' + name, output)




    
