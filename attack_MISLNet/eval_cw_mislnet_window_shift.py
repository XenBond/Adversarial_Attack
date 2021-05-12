import tensorflow as tf
from os.path import join
import sys, os
sys.path.append('../')

from data_pipe import TFRecordDataset
from MISLNet_Model import MISLNet_Model
from cleverhans.utils_tf import tf_model_load
from cleverhans.attacks import CarliniWagnerL2
import numpy as np
import PIL
import tqdm
import glob
import cv2
import image_generator

def accuracy_of_batch(predictions, labels):
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

T_LBS = 26 # target label
DB_NUM = 26
model_path = 'model-18'
im_folder = '/media/nas2/misl_image_db_clean/Canon PowerShot G10/1/'
patch_size = 256

def quadra_generator(im_folder, ext='jpg'):
    im_list = glob.glob(im_folder + '*.' + ext)
    for im_path in im_list:
        im = cv2.imread(im_path)
        shape = im.shape
        crop = im[int(shape[0] / 2 - patch_size) : int(shape[0] / 2 + patch_size),\
            int(shape[1] / 2 - patch_size) : int(shape[1] / 2 + patch_size)]
        l_up = crop[:patch_size, :patch_size]
        r_up = crop[:patch_size, patch_size:]
        l_down = crop[patch_size:, :patch_size]
        r_down = crop[patch_size:, patch_size:]
        output = np.array([l_up, r_up, l_down, r_down])
        yield output

im_generator = quadra_generator(im_folder, ext='JPG')

#'Motorola_X_val.tfrecords'
n_classes = 71
learning_rate = 1e-1
iterations = 100

image_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3], name='input_data')
label_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, n_classes], name='input_labels')
adv_ys=np.zeros((4, n_classes),  dtype=np.int32)
adv_ys[:, T_LBS]=1

model = MISLNet_Model('MISLNet_P', n_classes)
preds = model.get_logits(image_ph)
pred_labels = tf.argmax(preds, 1)
batch_accuracy=accuracy_of_batch(preds, label_ph)

with tf.Session() as sess:
    cw = CarliniWagnerL2(model, sess=sess)
    cw_params = {'binary_search_steps': 5,
            'y_target': adv_ys,
            'max_iterations': iterations,
            'learning_rate': learning_rate,
            'batch_size': 4,
            'clip_max': 255,
            'clip_min': 0,
            'initial_const': 1,
            'abort_early': True}

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    tf_model_load(sess, model_path)

    before_attack_batch_acc=0.
    after_attack_batch_acc=0.
    after_attack_batch_acc_org=0.
    validation_iters=10
    for ii in tqdm.tqdm(range(validation_iters)):
        iix_val = next(im_generator)
        print(iix_val.shape)
        org_preds = sess.run(pred_labels, feed_dict={image_ph: iix_val})
        print(org_preds)
        adv = cw.generate_np(iix_val, **cw_params)
        # Generate adversarial examples and return them as a NumPy array.
        percent_perturbed = np.mean(np.sum((adv - iix_val) ** 2, axis=(1, 2, 3)) ** .5)
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
        avg_perturb = np.mean(np.power(adv - iix_val, 2))
        print('Avg. L2 scores:', avg_perturb)
        attacked_images = adv
        attacked_images = attacked_images.astype(np.uint8)
        org_images = iix_val.astype(np.uint8)
        new_labels = sess.run(pred_labels, feed_dict={image_ph: attacked_images.astype(np.float32)})
        
        print(new_labels)
        
        new_patch = np.zeros((patch_size, patch_size, 3))
        new_patch[:128, :128] = attacked_images[0][128:, 128:]
        new_patch[:128, 128:] = attacked_images[1][128:, :128]
        new_patch[128:, :128] = attacked_images[2][:128, 128:]
        new_patch[128:, 128:] = attacked_images[3][:128, :128]
        shift_labels = sess.run(pred_labels, feed_dict={image_ph: new_patch[np.newaxis, ...].astype(np.float32)})
        
        print(shift_labels)



    
