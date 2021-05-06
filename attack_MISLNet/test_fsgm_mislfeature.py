import tensorflow as tf
from os.path import join
import sys, os
sys.path.append('../')

from data_pipe import TFRecordDataset
from MISLNet_Features import MISLNet_Features
from mislnet_model import MISLNet_pre
from cleverhans.utils_tf import tf_model_load
from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import MSE_loss
import numpy as np
import PIL
import tqdm
import glob
def accuracy_of_batch(predictions, labels):
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

T_LBS = 26 # target label
DB_NUM = 10
REF_NUM = 4
model_path = 'model-18'
data_dir = '/media/nas2/Shengbang/misl_image_db/'
tfrecord_list = glob.glob(data_dir + '*val.tfrecords')
#'Motorola_X_val.tfrecords'
n_classes = 71
patch_size = 256
batch_size = 32
source_samples = 100
n_epochs=6
learning_rate = 1
iterations = 1
ITER = 50 # iteration for single attack

image_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3], name='input_data')
label_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, n_classes], name='input_labels')

classify = MISLNet_pre(image_ph, n_classes, False, reuse=tf.AUTO_REUSE, namescope='MISLNet_P')
pred_lbs = tf.argmax(classify, 1)
batch_accuracy=accuracy_of_batch(classify, label_ph)

model = MISLNet_Features('MISLNet_P')
preds = model.get_logits(image_ph)

val_iter = TFRecordDataset(tfrecord_list[DB_NUM], batch_size=batch_size, crop_size=patch_size)
ref_iter = TFRecordDataset(tfrecord_list[REF_NUM], batch_size=batch_size, crop_size=patch_size)
val_image, val_label = val_iter.get_next()
ref_image, ref_label = ref_iter.get_next()
val_init = val_iter.initializer
ref_init = ref_iter.initializer

with tf.Session() as sess:
    var_list = [v for v in tf.global_variables() if 'MISLNet_P' in v.name]
    saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=None)
    saver.restore(sess, model_path)
    sess.run(val_init)
    sess.run(ref_init)
    fgsm = FastGradientMethod(model, sess=sess)
    iix_ref, iiy_ref = sess.run([ref_image, ref_label])
    references = sess.run(preds, feed_dict={image_ph: iix_ref})
    ref_mean = references[0]
    for ii in range(references.shape[0]):
        references[ii,:] = ref_mean
    print('ref size:', references.shape)
    fgsm_params = {'y_target': references,
            'loss_fn': MSE_loss,
            'eps': learning_rate,
            'ord': np.inf,
            'clip_max': 255,
            'clip_min': 0}

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    validation_iters= int(source_samples/batch_size)
    print('Step here')

    #tf_model_load(sess, model_path)

    before_attack_batch_acc=0.
    after_attack_batch_acc=0.
    after_attack_batch_acc_org=0.
    validation_iters=1
    for ii in tqdm.tqdm(range(validation_iters)):
        iix_val, iiy_val = sess.run([val_image, val_label])
        
        print(iix_val.shape, np.argmax(iiy_val, -1)[0])
        print(iix_ref.shape, np.argmax(iiy_ref, -1)[0])
        acc_batch_1 = sess.run(batch_accuracy, feed_dict={image_ph: iix_val, label_ph: iiy_val})
        before_attack_batch_acc += acc_batch_1
        print('init acc:', acc_batch_1)
        alpha = 5
        adv = iix_val.copy()
        for kk in tqdm.tqdm(range(ITER)):
            _, grad = fgsm.generate_np(adv, **fgsm_params)
            peak = np.amax(np.abs(grad), axis=(1, 2, 3))
            diff = grad
            for jj in range(peak.shape[0]):
                diff[jj] = grad[jj] / peak[jj] 
            adv = iix_val + alpha * diff
            adv = np.clip(adv, 0, 255)
        adv.astype(np.uint8)
        # Generate adversarial examples and return them as a NumPy array.
        percent_perturbed = np.mean(np.sum((adv - iix_val) ** 2, axis=(1, 2, 3)) ** .5)
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
        avg_perturb = np.mean(np.power(adv - iix_val, 2))
        print('Avg. L2 scores:', avg_perturb)
        attacked_images = adv
        attacked_images = attacked_images.astype(np.uint8)
        org_images = iix_val.astype(np.uint8)
        org_label = sess.run(pred_lbs, feed_dict={image_ph: iix_val})
        atk_label = sess.run(pred_lbs, feed_dict={image_ph: adv})
        print('org label', org_label)
        print('atk label', atk_label)
        acc_batch_2 = sess.run(batch_accuracy, feed_dict={image_ph: adv, label_ph: iiy_ref})
        acc_batch_3 = sess.run(batch_accuracy, feed_dict={image_ph: adv, label_ph: iiy_val})
        after_attack_batch_acc += acc_batch_2
        after_attack_batch_acc_org += acc_batch_3
        '''
        for ii in range(attacked_images.shape[0]):
            sample_attacked = PIL.Image.fromarray(attacked_images[ii])
            sample_org = PIL.Image.fromarray(org_images[ii])
            sample_attacked.save('./results/Carlini-Wagner/attack_'+str(ii) +'.png','png')
            sample_org.save('./results/Carlini-Wagner/org_'+str(ii) +'.png','png')
        '''
    print("Accuracy over val set before attack: ", before_attack_batch_acc /validation_iters)
    print("Accuracy over val set after attack on org label: ", after_attack_batch_acc_org /validation_iters)
    print("Accuracy over val set after attack: ", after_attack_batch_acc /validation_iters)




    
