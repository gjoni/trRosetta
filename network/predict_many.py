import warnings, logging, os, sys
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import tensorflow as tf
from utils import *
from arguments import get_args_many

args = get_args_many()

MDIR         = args.MDIR

n2d_layers   = 61
n2d_filters  = 64
window2d     = 3
wmin         = 0.8
ns           = 21


# load network weights in RAM
w,b,beta_,gamma_ = load_weights(args.MDIR)

#
# network
#
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(allow_growth=True)
)

activation = tf.nn.elu
conv1d = tf.layers.conv1d
conv2d = tf.layers.conv2d
with tf.Graph().as_default():

    with tf.name_scope('input'):
        ncol = tf.placeholder(dtype=tf.int32, shape=())
        nrow = tf.placeholder(dtype=tf.int32, shape=())
        msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))

    #
    # collect features
    #
    msa1hot  = tf.one_hot(msa, ns, dtype=tf.float32)
    weights = reweight(msa1hot, wmin)

    # 1D features
    f1d_seq = msa1hot[0,:,:20]
    f1d_pssm = msa2pssm(msa1hot, weights)

    f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1)
    f1d = tf.expand_dims(f1d, axis=0)
    f1d = tf.reshape(f1d, [1,ncol,42])

    # 2D features
    f2d_dca = tf.cond(nrow>1, lambda: fast_dca(msa1hot, weights), lambda: tf.zeros([ncol,ncol,442], tf.float32))
    f2d_dca = tf.expand_dims(f2d_dca, axis=0)

    f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]), 
                    tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                    f2d_dca], axis=-1)
    f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])


    #
    # 2D network
    #

    # store ensemble of networks in separate branches
    layers2d = [[] for _ in range(len(w))]
    preds = [[] for _ in range(4)]

    Activation   = tf.nn.elu

    for i in range(len(w)):

        layers2d[i].append(Conv2d(f2d,w[i][0],b[i][0]))
        layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][0],gamma_[i][0]))
        layers2d[i].append(Activation(layers2d[i][-1]))

        # resnet
        idx = 1
        dilation = 1
        for _ in range(n2d_layers):
            layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
            layers2d[i].append(Activation(layers2d[i][-1]))
            idx += 1
            layers2d[i].append(Conv2d(layers2d[i][-1],w[i][idx],b[i][idx],dilation))
            layers2d[i].append(InstanceNorm(layers2d[i][-1],beta_[i][idx],gamma_[i][idx]))
            layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))
            idx += 1
            dilation *= 2
            if dilation > 16:
                dilation = 1


        # probabilities for theta and phi
        preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][123],b[i][123]))[0])
        preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][124],b[i][124]))[0])

        # symmetrize
        layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

        # probabilities for dist and omega
        preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][125],b[i][125]))[0])
        preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][127],b[i][127]))[0])
        #preds[4].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][126],b[i][126]))[0])

    # average over all branches
    prob_theta = tf.reduce_mean(tf.stack(preds[0]),axis=0)
    prob_phi   = tf.reduce_mean(tf.stack(preds[1]),axis=0)
    prob_dist  = tf.reduce_mean(tf.stack(preds[2]),axis=0)
    prob_omega = tf.reduce_mean(tf.stack(preds[3]),axis=0)


    with tf.Session(config=config) as sess:

        # loop over all A3M files in the imput folder
        for filename in os.listdir(args.ALNDIR):
            if not filename.endswith(".a3m"):
                continue

            # parse & predict
            a3m = parse_a3m(args.ALNDIR + '/' + filename)
            print("processing:", filename)

            pd, pt, pp, po = sess.run([prob_dist, prob_theta, prob_phi, prob_omega],
                                    feed_dict = {msa : a3m, ncol : a3m.shape[1], nrow : a3m.shape[0] })

	    # save distograms & anglegrams
            npz_file = args.NPZDIR + '/' + filename[:-3] + 'npz'
            np.savez_compressed(npz_file, dist=pd, omega=po, theta=pt, phi=pp)

