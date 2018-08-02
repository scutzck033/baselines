import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    # h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h2)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def dqn_cnn(scaled_images, **conv_kwargs):

    activ = tf.nn.leaky_relu
    c1 = activ(conv(scaled_images, 'c1', nf=32, rf=5, stride=1, init_scale=np.sqrt(2),
                   **conv_kwargs))
    # p1 = tf.nn.max_pool(c1,[1,2,2,1],[1,1,1,1],padding='SAME',name='p1')
    c2 = activ(conv(c1, 'c2', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    # p2 = tf.nn.max_pool(c2, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME', name='p2')
    c3 = activ(conv(c2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h1 = conv_to_fc(c3)
    h1 = activ(fc(h1, 'fc1', nh=256, init_scale=np.sqrt(2)))
    return activ(fc(h1, 'fc2', nh=128, init_scale=np.sqrt(2)))

def lenet_cnn(scaled_images, **conv_kwargs):
    activ = tf.nn.leaky_relu
    c1 = activ(conv(scaled_images, 'c1', nf=6, rf=5, stride=1, init_scale=np.sqrt(2),
                    **conv_kwargs))
    p1 = tf.nn.max_pool(c1, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME', name='p1')
    c2 = activ(conv(p1, 'c2', nf=16, rf=5, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    p2 = tf.nn.max_pool(c2, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME', name='p2')
    c3 = activ(conv(p2, 'c3', nf=120, rf=5, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h1 = conv_to_fc(c3)
    return activ(fc(h1, 'fc1', nh=128, init_scale=np.sqrt(2)))


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)
        with tf.variable_scope("model", reuse=reuse):
            h = dqn_cnn(processed_x, **conv_kwargs)
            vf = fc(h, 'v', 1)[:,0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X, processed_x = observation_input(ob_space, nbatch)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)


        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

class twoStreamCNNPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs): #pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        img_X, processed_img_x = observation_input(ob_space[0], nbatch)
        vec_X, processed_vec_x = observation_input(ob_space[1], nbatch)
        with tf.variable_scope("model", reuse=reuse):
            # img feature extractor
            img_h = lenet_cnn(processed_img_x, **conv_kwargs)

            # vec feature extractor
            activ = tf.nn.leaky_relu

            vec_h1 = activ(fc(processed_vec_x, 'pi_fc1', nh=32, init_scale=np.sqrt(2)))
            vec_h = activ(fc(vec_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))

            # feature concat
            h = tf.concat([img_h,vec_h],1)

            vf = fc(h, 'v', 1)[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            img_x_input = ob[0]
            vec_x_input = ob[1]

            a, v, neglogp = sess.run([a0, vf, neglogp0], {img_X:img_x_input,vec_X:vec_x_input})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            img_x_input = ob[0]
            vec_x_input = ob[1]

            return sess.run(vf, {img_X:img_x_input,vec_X:vec_x_input})

        self.img_X = img_X
        self.vec_X = vec_X
        self.vf = vf
        self.step = step
        self.value = value