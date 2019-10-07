import os,sys
import json
import tensorflow as tf
from utils import *
from arguments import *

args = get_args()

msa_file     = args.ALN
npz_file     = args.NPZ

MDIR         = args.MDIR

n2d_layers   = 61
n2d_filters  = 64
window2d     = 3
wmin         = 0.8
ns           = 21

a3m = parse_a3m(msa_file)

contacts = {'pd':[], 'po':[], 'pt':[], 'pp':[]}



#
# network
#
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
activation = tf.nn.elu
conv1d = tf.layers.conv1d
conv2d = tf.layers.conv2d
with tf.Graph().as_default():

    with tf.name_scope('input'):
        ncol = tf.placeholder(dtype=tf.int32, shape=())
        nrow = tf.placeholder(dtype=tf.int32, shape=())
        msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))
        is_train = tf.placeholder(tf.bool, name='is_train')

    #
    # collect features
    #
    msa1hot  = tf.one_hot(msa, ns, dtype=tf.float32)
    w = reweight(msa1hot, wmin)

    # 1D features
    f1d_seq = msa1hot[0,:,:20]
    f1d_pssm = msa2pssm(msa1hot, w)

    f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
    f1d = tf.expand_dims(f1d, axis=0)
    f1d = tf.reshape(f1d, [1,ncol,42])

    # 2D features
    f2d_dca = tf.cond(nrow>1, lambda: fast_dca(msa1hot, w), lambda: tf.zeros([ncol,ncol,442], tf.float32))
    f2d_dca = tf.expand_dims(f2d_dca, axis=0)

    f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]), 
                    tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                    f2d_dca], axis=-1)
    f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])


    #
    # 2D network
    #
    layers2d = [f2d]
    layers2d.append(conv2d(layers2d[-1], n2d_filters, 1, padding='SAME'))
    layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
    layers2d.append(activation(layers2d[-1]))

    # stack of residual blocks with dilations
    dilation = 1
    for _ in range(n2d_layers):
        layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
        layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
        layers2d.append(activation(layers2d[-1]))
        layers2d.append(tf.keras.layers.Dropout(rate=0.15)(layers2d[-1], training=is_train))
        layers2d.append(conv2d(layers2d[-1], n2d_filters, window2d, padding='SAME', dilation_rate=dilation))
        layers2d.append(tf.contrib.layers.instance_norm(layers2d[-1]))
        layers2d.append(activation(layers2d[-1] + layers2d[-7]))
        dilation *= 2
        if dilation > 16:
            dilation = 1

    # anglegrams for theta
    logits_theta = conv2d(layers2d[-1], 25, 1, padding='SAME')
    prob_theta = tf.nn.softmax(logits_theta)

    # anglegrams for phi
    logits_phi = conv2d(layers2d[-1], 13, 1, padding='SAME')
    prob_phi = tf.nn.softmax(logits_phi)

    # symmetrize
    layers2d.append(0.5 * (layers2d[-1] + tf.transpose(layers2d[-1], perm=[0,2,1,3])))

    # distograms
    logits_dist = conv2d(layers2d[-1], 37, 1, padding='SAME')
    prob_dist = tf.nn.softmax(logits_dist)

    # beta-strand pairings (not used)
    logits_bb = conv2d(layers2d[-1], 3, 1, padding='SAME')
    prob_bb = tf.nn.softmax(logits_bb)

    # anglegrams for omega
    logits_omega = conv2d(layers2d[-1], 25, 1, padding='SAME')
    prob_omega = tf.nn.softmax(logits_omega)

    saver = tf.train.Saver()

    #for ckpt in ['model.xaa', 'model.xab', 'model.xac', 'model.xad', 'model.xae']:
    for filename in os.listdir(MDIR):
        if not filename.endswith(".index"):
            continue
        ckpt = MDIR+"/"+os.path.splitext(filename)[0]
        with tf.Session(config=config) as sess:
            saver.restore(sess, ckpt)
            pd, pt, pp, po = sess.run([prob_dist, prob_theta, prob_phi, prob_omega],
                                       feed_dict = {msa : a3m, ncol : a3m.shape[1], nrow : a3m.shape[0], is_train : 0})
            contacts['pd'].append(pd[0])
            contacts['pt'].append(pt[0])
            contacts['po'].append(po[0])
            contacts['pp'].append(pp[0])
            print(ckpt, '- done')

# average over all network params
contacts['pd'] = np.mean(contacts['pd'], axis=0)
contacts['pt'] = np.mean(contacts['pt'], axis=0)
contacts['po'] = np.mean(contacts['po'], axis=0)
contacts['pp'] = np.mean(contacts['pp'], axis=0)

# save distograms & anglegrams
np.savez_compressed(npz_file, dist=contacts['pd'], omega=contacts['po'], theta=contacts['pt'], phi=contacts['pp'])

