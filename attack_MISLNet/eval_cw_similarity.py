import tensorflow as tf
from os.path import join
import sys, os
sys.path.append('../')

from data_pipe import TFRecordDataset
from CompareNet_Model import CompareNet_Model
from cleverhans.utils_tf import tf_model_load_namescope
from cleverhans.attacks import CarliniWagnerL2
import numpy as np
import PIL
import tqdm
import glob

def accuracy_of_batch(predictions, labels):
    correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy

DB_NUM = 2
REF_NUM = 4
model_path = './cam_128/-30'
data_dir = '/media/nas2/Shengbang/misl_image_db/'
tfrecord_list = glob.glob(data_dir + '*val.tfrecords')
print('dataset:', tfrecord_list[DB_NUM])
print('target:', tfrecord_list[REF_NUM])
#'Motorola_X_val.tfrecords'
patch_size = 128
batch_size = 16
source_samples = 100
n_epochs=6
learning_rate = 1e-2
iterations = 100

image_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, patch_size, patch_size, 3], name='input_data')
tlbs = np.array([[0, 1] for ii in range(batch_size)])
label = tf.constant(tlbs, dtype=tf.int32, shape=[batch_size, 2], name='input_labels')

val_iter = TFRecordDataset(tfrecord_list[DB_NUM], batch_size=batch_size, crop_size=patch_size)
val_image, val_label = val_iter.get_next()
val_init = val_iter.initializer

ref_iter = TFRecordDataset(tfrecord_list[REF_NUM], batch_size=batch_size, crop_size=patch_size)
ref_image, ref_label = ref_iter.get_next()
ref_init = ref_iter.initializer

with tf.Session() as sess:
    sess.run(ref_init)
    reference = sess.run(ref_image)
    for ii in range(reference.shape[0]):
        reference[ii] = reference[0]
print('ref size:', reference.shape)
model = CompareNet_Model('Similarity', batch_size, reference)
preds = model.get_logits(image_ph)
pred_lbs = tf.argmax(preds, 1)
batch_accuracy=accuracy_of_batch(preds, label)


with tf.Session() as sess:
    cw = CarliniWagnerL2(model, sess=sess)
    cw_params = {'binary_search_steps': 1,
            'y_target': tlbs,
            'max_iterations': iterations,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'clip_max': 255,
            'clip_min': 0,
            'initial_const': 1,
            'abort_early': True}

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    validation_iters= int(source_samples/batch_size)
    
    sess.run(val_init)
    sess.run(ref_init)

    tf_model_load_namescope(sess, model_path, local_namescope='Similarity/', model_namescope='')

    before_attack_batch_acc=0.
    after_attack_batch_acc=0.
    after_attack_batch_acc_org=0.
    validation_iters=1

    for ii in tqdm.tqdm(range(validation_iters)):
        iix_val = sess.run(val_image)
        iiy_val = sess.run(ref_image)
        print(iix_val.shape)
        predicted_labels, acc_batch_1 = sess.run([pred_lbs, batch_accuracy], feed_dict={image_ph: iix_val})
        before_attack_batch_acc += acc_batch_1
        print('init acc:', acc_batch_1)
        # debug
        
        adv = cw.generate_np(iix_val, **cw_params)
        adv = adv.astype(np.uint8)
        # Generate adversarial examples and return them as a NumPy array.
        percent_perturbed = np.mean(np.sum((adv - iix_val) ** 2, axis=(1, 2, 3)) ** .5)
        print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
        avg_perturb = np.mean(np.power(adv - iix_val, 2))
        print('Avg. L2 scores:', avg_perturb)
        attack_images = adv.astype(np.float32)
        print('init label:', predicted_labels)
        predicted_labels, acc_batch_2 = sess.run([pred_lbs, batch_accuracy], feed_dict={image_ph: attack_images, model.reference:iiy_val})
        print('attack label:', predicted_labels)
        after_attack_batch_acc += acc_batch_2
        '''
        try:
            os.mkdir('./results/FGSM/')
        except:
            pass
        for ii in range(attacked_images.shape[0]):
            sample_attacked = PIL.Image.fromarray(attacked_images[ii])
            sample_org = PIL.Image.fromarray(org_images[ii])
            sample_attacked.save('./results/FGSM/attack_'+str(ii) +'.png','png')
            sample_org.save('./results/FGSM/org_'+str(ii) +'.png','png')
        '''
    print("Accuracy over val set before attack: ", before_attack_batch_acc /validation_iters)
    print("Accuracy over val set after attack: ", after_attack_batch_acc /validation_iters)




    
