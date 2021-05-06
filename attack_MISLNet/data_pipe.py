import tensorflow as tf
import random
import sys
#Little utilities for helping with tfrecords
def _int64_feature(value):
    '''Force a feature/lable/etc. to int64
    just here to make tfrecord storage cleaner'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    '''Force a feature/lable/etc. to bytes
    just here to make tfrecord storage cleaner'''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#Map function for use with datasets. BH creation
def unpack(xmple,height=256,width=256,n_channels=3,crop_size=None):
    '''Not sure if an argument can actually be passed. Edit
    function for every experiment? Unpacks a tf.databse into
    an image and a label. For use in a .map() function'''
    features = tf.io.parse_single_example(xmple,
        features={'label': tf.io.FixedLenFeature([], tf.int64),
        'patch': tf.io.FixedLenFeature([], tf.string)})
    #Fill in the object with the features from the file
    image = tf.decode_raw(features['patch'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.squeeze(tf.reshape(image, [-1, height, width, n_channels]))
    if crop_size is not None:
        image = image[:crop_size,:crop_size]
    #label = tf.cast(features['label'], tf.int32)
    label = tf.cast(tf.one_hot(features['label'], 71), tf.int32)
    return image, label

def TFRecordDataset(fname, batch_size=16, dims=[256,256,3], rpt=True, shuffbuff=200, crop_size=None):
    dataset = tf.data.TFRecordDataset(fname).map(lambda x: unpack(x,dims[0],dims[1],dims[2],crop_size))
    if(rpt is True):
        dataset = dataset.repeat()
    if(shuffbuff is not None):
        dataset = dataset.shuffle(shuffbuff)
    if(batch_size is not None):
        dataset = dataset.batch(batch_size)
    dataset = tf.compat.v1.data.make_initializable_iterator(dataset)
    return dataset


#Create an iterable dataset from BH video tfrecords
def TFdatasetIterator(fname,batch_size=16,dims=[256,256,3],rpt=True,shuffbuff=2000,crop_size=None, interleave=False):
    '''Open a tfrecords file, shuffles, batches, and returns an iterator'''
    #Determind if we're reading from multiple datasets
    #if isinstance(fname,(list,tuple)) and len(fname)>1:
    if interleave is True:
        fname = tf.strings.split(fname, sep=',').values
        #totalPatches = sum(_num_patches(fnames) for fnames in fname)
        dataset = (tf.data.Dataset.from_tensor_slices(fname).interleave(
            lambda x:tf.data.TFRecordDataset(x).map(lambda x: unpack(x,dims[0],dims[1],dims[2],crop_size),
            num_parallel_calls=8),cycle_length=tf.cast(tf.size(fname), tf.int64),block_length=1))
    else:
        if isinstance(fname,(list,tuple)):
            fname=fname[0]
        print('not tuple!', fname)
        #totalPatches = _num_patches(fname)
        dataset = tf.data.TFRecordDataset([fname],num_parallel_reads=4)
        dataset = dataset.map(lambda x: unpack(x,dims[0],dims[1],dims[2],crop_size))
    #shuffle and perform other operations if necessary
    if(shuffbuff is not None):
        dataset = dataset.shuffle(shuffbuff).prefetch(200)
    # repeat must be in front of batch.
    dataset = dataset.repeat()
    if batch_size>1:
        dataset = dataset.batch(batch_size)
    #dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/gpu:0',150))
    dataset = tf.compat.v1.data.make_initializable_iterator(dataset)
    return dataset#, totalPatches


def cartesian_product(a,b): #all combinations of elements in a and b, with a first and b second
    #From jdehesa in https://stackoverflow.com/questions/53123763/how-to-perform-cartesian-product-with-tensorflow
    c = tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1)
    c = tf.reshape(c, (-1, 2))
    return c
    
def combinations(a,b):
    #all unique combinations of elements in a and b
    #order matters (e.g. [1 2] is considered unique from [2 1], so we get both)
    #removes any instances where a and b are the same. we shouldn't ever encounter a scenario where we need to know whether the same patch is forensically same/different
    #tested with single values (indices), not sure if it would work on more complex tensors, though it should
    
    c1 = cartesian_product(a,b)
    c2 = cartesian_product(b,a)
    c = tf.concat([c1,c2],0)
    
    inds_diff = tf.where(tf.not_equal(c[:,1], #first and second column are different
    				  c[:,0]))
    inds_diff = tf.squeeze(inds_diff) #remove size 1 dimensions
    out = tf.gather(c,inds_diff) #return only combinations that are different
    
    return out


def paired_shuffle(x1,x2): #not used, but keeping just in case. 
    #shuffle x1 and x2 in the same order
    #useful for shuffling samples and associated labels
    
    inds = tf.range(tf.shape(x1)[0]) #indices, assume x1 and x2 have same dim 0
    rnds = tf.random.shuffle(inds) #randomly shuffle the indices
    x1_shuffled = tf.gather(x1,rnds) #shuffle x1 and x2
    x2_shuffled = tf.gather(x2,rnds)
    
    return x1_shuffled, x2_shuffled

def random_pairings_two_source(X_0, y_0, X_1, y_1, n_same=24, n_diff=24):
    # Diff version of random_pairings. For each pair, first item always from 1st source and 2nd item always from 2nd source. 
    X = tf.concat([X_0, X_1], 0)
    y = tf.concat([y_0, y_1], 0)
    
    inds_1 = tf.range(tf.shape(y_0)[0])
    inds_2 = tf.range(tf.shape(y_1)[0], tf.shape(y_1)[0] * 2)
    
    icombs = cartesian_product(inds_1, inds_2)
    ycombs = tf.gather(y, icombs)
    
    inds_diff = tf.where(tf.not_equal(ycombs[:, 1], ycombs[:, 0]))
    inds_diff = tf.squeeze(inds_diff)
    n_diff_available = tf.shape(inds_diff)[0]
    
    inds_same = tf.where(tf.equal(ycombs[:, 1], ycombs[:, 0]))
    inds_same = tf.squeeze(inds_same)
    n_same_available = tf.shape(inds_same)[0]
    
    rinds_same = tf.random.shuffle(inds_same)[:n_same]
    rinds_diff = tf.random.shuffle(inds_diff)[:n_diff]
    #rinds_same = inds_same[:n_same]
    #rinds_diff = inds_diff[:n_diff]
    rinds = tf.concat([rinds_same, rinds_diff],0) #combine indices of randomly chosen same and different combinations
    ricombs = tf.gather(icombs, rinds) #convert to pairs of X/y indices
    
    X_pair = tf.gather(X,ricombs) #gather patch pairs
    labels_pair = tf.gather(y,ricombs) #gather label pairs
    y_out = tf.equal(labels_pair[:,1], labels_pair[:,0]) #tell me when the pairs have the same label
    y_out = tf.cast(y_out,tf.int32) #same label = 1, different label = 0
    
    return X_pair, labels_pair, y_out, n_same_available, n_diff_available
    

def random_pairings(X,y,n_same=24,n_diff=24):

    #indices of the patches
    inds = tf.range(tf.shape(y)[0])
    
    #all pairings of indices, and corresponding class labels y (with no pairings of the same patch)
    icombs = combinations(inds,inds)
    ycombs = tf.gather(y,icombs)
    #indices of same and different camera model
    inds_diff = tf.where(tf.not_equal(ycombs[:,1], #first and second column are different
    				  ycombs[:,0]))
    inds_diff = tf.squeeze(inds_diff) #remove size 1 dimensions
    n_diff_available = tf.shape(inds_diff)[0] #number of available "different" pairs
    
    inds_same = tf.where(tf.equal(ycombs[:,1], #first and second column are the same
    			      ycombs[:,0]))
    inds_same = tf.squeeze(inds_same) #remove size 1 dimensions
    n_same_available = tf.shape(inds_same)[0] #number of available "same" pairs
    
    #randomly select n combinations. How do we make sure there are enough 
    #combinations/n isn't too big?. I guess right now we need to be careful to choose a batch size that is big enough.
    rinds_same = tf.random.shuffle(inds_same)[:n_same]
    rinds_diff = tf.random.shuffle(inds_diff)[:n_diff]
    rinds = tf.concat([rinds_same,rinds_diff],0) #combine indices of randomly chosen same and different combinations
    ricombs = tf.gather(icombs,rinds) #convert to pairs of X/y indices
    
    X_pair = tf.gather(X,ricombs) #gather patch pairs
    labels_pair = tf.gather(y,ricombs) #gather label pairs
    y_out = tf.equal(labels_pair[:,1], labels_pair[:,0]) #tell me when the pairs have the same label
    y_out = tf.cast(y_out,tf.int32) #same label = 1, different label = 0
    
    return X_pair, labels_pair, y_out, n_same_available, n_diff_available


	
