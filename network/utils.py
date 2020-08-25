import os
import numpy as np
import string
import tensorflow as tf

# read A3M and convert letters into
# integers in the 0..20 range
def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa


# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum( -f_i * tf.math.log(f_i), axis=1)
    return tf.concat([f_i, h_i[:,None]], axis=1)


# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w


# shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty = 4.5):

    nr = tf.shape(msa1hot)[0]
    nc = tf.shape(msa1hot)[1]
    ns = tf.shape(msa1hot)[2]

    with tf.name_scope('covariance'):
        x = tf.reshape(msa1hot, (nr, nc * ns))
        num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
        mean = tf.reduce_sum(x * weights[:,None], axis=0, keepdims=True) / num_points
        x = (x - mean) * tf.sqrt(weights[:,None])
        cov = tf.matmul(tf.transpose(x), x)/num_points

    with tf.name_scope('inv_convariance'):
        cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(weights))
        inv_cov = tf.linalg.inv(cov_reg)
        
        x1 = tf.reshape(inv_cov,(nc, ns, nc, ns))
        x2 = tf.transpose(x1, [0,2,1,3])
        features = tf.reshape(x2, (nc, nc, ns * ns))
        
        x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(nc))
        apc = tf.reduce_sum(x3,0,keepdims=True) * tf.reduce_sum(x3,1,keepdims=True) / tf.reduce_sum(x3)
        contacts = (x3 - apc) * (1-tf.eye(nc))

    return tf.concat([features, contacts[:,:,None]], axis=2)


def load_weights(DIR):

    # load networks in RAM
    w,b = [],[]
    beta_,gamma_ = [],[]

    for filename in os.listdir(DIR):
        if not filename.endswith(".index"):
            continue
        mname = DIR+"/"+os.path.splitext(filename)[0]
        print('reading weights from:', mname)

        w.append([
            tf.train.load_variable(mname, 'conv2d/kernel')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
            for i in range(128)])

        b.append([
            tf.train.load_variable(mname, 'conv2d/bias')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
            for i in range(128)])

        beta_.append([
            tf.train.load_variable(mname, 'InstanceNorm/beta')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
            for i in range(123)])

        gamma_.append([
            tf.train.load_variable(mname, 'InstanceNorm/gamma')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
            for i in range(123)])

    return w,b,beta_,gamma_


def InstanceNorm(features,beta,gamma):
    mean,var = tf.nn.moments(features,axes=[1,2])
    x = (features - mean[:,None,None,:]) / tf.sqrt(var[:,None,None,:]+1e-5)
    out = tf.constant(gamma)[None,None,None,:]*x + tf.constant(beta)[None,None,None,:]
    return out


def Conv2d(features,w,b,d=1):
    x = tf.nn.conv2d(features,tf.constant(w),strides=[1,1,1,1],padding="SAME",dilations=[1,d,d,1]) + tf.constant(b)[None,None,None,:]
    return x

