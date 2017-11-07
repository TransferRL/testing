""" An rbm implementation for TensorFlow, based closely on the one in Theano """

import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm


def sample_prob(probs):
    """Takes a tensor of probabilities (as from a sigmoidal activation)
       and samples from all the distributions"""
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(probs.get_shape())))


def generate_v(n_v, mu_v, sig_v):
    v = np.random.normal(mu_v,sig_v,n_v)
    return v


class RBM(object):
    """ represents a 3-way rbm """

    def __init__(self, name, v1_size, h_size, v2_size, n_samples, num_epochs=100, learning_rate=0.1, k=1,
                 persistent=False, use_tqdm=True):
        with tf.name_scope("rbm_" + name):
            self.v1_size = v1_size
            self.v2_size = v2_size
            self.h_size = h_size
            self.weights = tf.Variable(
                tf.truncated_normal([v1_size, h_size, v2_size],
                                    stddev=1.0 / math.sqrt(float((v1_size+v2_size)/2))), name="weights")
            self.h_bias = tf.Variable(tf.zeros([1,h_size]), name="h_bias")
            self.v1_bias = tf.Variable(tf.zeros([1,v1_size]), name="v1_bias")
            self.v1_var = tf.constant(np.ones([v1_size]), name="v1_var")
            self.v2_bias = tf.Variable(tf.zeros([1,v2_size]), name="v1_bias")
            self.v2_var = tf.constant(np.ones([v2_size]), name="v1_var")

            self.n_samples = n_samples

            self.chain_h = None
            self.chain_v1 = None
            self.chain_v2 = None

            self.upd_w = None
            self.upd_hb = None
            self.upd_v1b = None
            self.upd_v2b = None

            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.k = k
            self.persistent = persistent

            self.use_tqdm = use_tqdm

            self.v1_input = tf.placeholder('float32', (self.n_samples, self.v1_size))
            self.v2_input = tf.placeholder('float32', (self.n_samples, self.v2_size))

            self.tf_session = None

    def _prop_helper(self, a, b, perm):
        """ perm specifies matrix orientation"""
        wt = tf.transpose(self.weights,perm=perm)
        wtv = tf.einsum('ijk,kl->ijl',wt,tf.transpose(a))
        wtvt = tf.transpose(wtv,perm=[2,0,1])
        wtvtv = tf.matmul(wtvt,tf.expand_dims(b,axis=-1))
        wtvtvs = tf.reduce_sum(wtvtv,axis=-1)
        return wtvtvs

    def prop_v1v2_h(self, v1, v2, v_size):
        """ P(h|v1,v2) """
        return tf.nn.sigmoid(self._prop_helper(v1, v2, [1, 2, 0]) + tf.tile(self.h_bias,[v_size,1]))

    def prop_v1h_v2(self, v1, h, v_size):
        """ P(v2|v1,h) """
        return self._prop_helper(v1, h, [2, 1, 0]) + tf.tile(self.v2_bias,[v_size,1])

    def prop_v2h_v1(self, v2, h, v_size):
        """ P(v1|v2,h) """
        return self._prop_helper(v2, h, [0, 1, 2]) + tf.tile(self.v1_bias,[v_size,1])

    def sample_v1_given_v2h(self, v2, h):
        """ generate sample of v1 from v2 and h"""
        dist = tf.contrib.distributions.Normal(tf.cast(self.prop_v2h_v1(v2, h, v2.shape[0]),tf.float32), tf.cast(tf.tile(tf.expand_dims(self.v1_var,0),[v2.shape[0],1]),tf.float32))
        return tf.reduce_sum(dist.sample(1),0)

    def sample_v2_given_v1h(self, v1, h):
        """ generate sample of v1 from v2 and h"""
        dist = tf.contrib.distributions.Normal(tf.cast(self.prop_v1h_v2(v1, h, v1.shape[0]),tf.float32), tf.cast(tf.tile(tf.expand_dims(self.v2_var,0), [v1.shape[0],1]),tf.float32))
        return tf.reduce_sum(dist.sample(1),0)

    def sample_h_given_v1v2(self, v1, v2):
        """ Generate a sample from the hidden layer """
        return sample_prob(self.prop_v1v2_h(v1,v2,v1.shape[0]))

    @staticmethod
    def get_delta_products(v1, h, v2):
        v1 = tf.expand_dims(v1, axis=-1)
        h = tf.expand_dims(h, axis=-1)
        v2 = tf.expand_dims(v2, axis=-1)
        prod = tf.matmul(v1, tf.transpose(h, perm=[0,2,1]))
        prod = tf.transpose(tf.expand_dims(prod, -1), perm=[1, 0, 2, 3])
        prod = tf.einsum('ijkl,jmn->ijkn', prod, tf.transpose(v2, perm=[0, 2, 1]))
        prod = tf.transpose(prod, perm=[1, 0, 2, 3])
        return tf.reduce_mean(prod, axis=0)

    def gibbs(self, v1, h, v2, n_samples):

        # using activations
        v1 = self.prop_v2h_v1(v2, h, n_samples)
        v2 = self.prop_v1h_v2(v1, h, n_samples)
        h = self.prop_v1v2_h(v1, v2, n_samples)

        # using sampling
        #v1 = self.sample_v1_given_v2h(v2, h)
        #v2 = self.sample_v2_given_v1h(v1, h)
        #h = sample_h_given_v1v2(v1, v2)

        return v1, h, v2

    def train(self, v1_input, v2_input):
        """train RBM"""

        self.pcd_k()
        with tf.Session() as self.tf_session:
            init = tf.global_variables_initializer()
            self.tf_session.run(init)

            pbar = tqdm(range(self.num_epochs))
            for i in pbar:
                self.one_train_step(v1_input, v2_input)
                # err = self.tf_session.run(self.reconstruction_error(),
                #                           feed_dict={self.v1_input: v1_input, self.v2_input: v2_input})
                # pbar.set_description('squared reconstruction error: {}'.format(0))

            err = self.tf_session.run(self.reconstruction_error(),
                                      feed_dict={self.v1_input: v1_input, self.v2_input: v2_input})
            print('cost:{}'.format(err))

    def one_train_step(self, v1_sample, v2_sample):
        """run one training step"""

        # TODO: implement batches

        updates = [self.upd_w, self.upd_v1b, self.upd_v2b, self.upd_hb]
        self.tf_session.run(updates, feed_dict={self.v1_input: v1_sample, self.v2_input: v2_sample})

    def pcd_k(self):
        "k-step (persistent) contrastive divergence"

        if self.chain_v1 is None and self.persistent:
            self.chain_v1 = self.v1_input
        if self.chain_v2 is None and self.persistent:
            self.chain_v2 = self.v2_input
        if self.chain_h is None and self.persistent:
            self.chain_h = self.prop_v1v2_h(self.chain_v1, self.chain_v2, self.n_samples)

        mcmc_v1, mcmc_v2 = (self.chain_v1, self.chain_v2) if self.persistent else (self.v1_input, self.v2_input)

        start_h = self.chain_h if self.persistent else self.prop_v1v2_h(self.v1_input, self.v2_input, self.n_samples)
        mcmc_h = start_h

        for n in range(self.k):
            mcmc_v1, mcmc_h, mcmc_v2 = self.gibbs(mcmc_v1, mcmc_h, mcmc_v2, self.n_samples)

        if self.persistent:
            self.chain_v1, self.chain_h, self.chain_v2 = mcmc_v1, mcmc_h, mcmc_v2

        w_positive_grad = self.get_delta_products(self.v1_input, start_h, self.v2_input)
        w_negative_grad = self.get_delta_products(mcmc_v1, mcmc_h, mcmc_v2)

        self.upd_w = self.weights.assign_add(self.learning_rate * (w_positive_grad - w_negative_grad))

        self.upd_v1b = self.v1_bias.assign_add(self.learning_rate * tf.reduce_mean(self.v1_input - mcmc_v1, 0,
                                                                                   keep_dims=True))
        self.upd_v2b = self.v2_bias.assign_add(self.learning_rate * tf.reduce_mean(self.v2_input - mcmc_v2, 0,
                                                                                   keep_dims=True))

        self.upd_hb = self.h_bias.assign_add(self.learning_rate * tf.reduce_mean(start_h - mcmc_h, 0, keep_dims=True))

    def reconstruction_error(self):
        """ The one-step reconstruction cost for both visible layers """
        h = self.prop_v1v2_h(self.v1_input, self.v2_input, self.v1_input.shape[0])

        v1_err = tf.cast(self.v1_input, tf.float32) - self.sample_v1_given_v2h(self.v2_input, h)
        v1_err = tf.reduce_sum(v1_err * v1_err, [0, 1])

        v2_err = tf.cast(self.v2_input,tf.float32) - self.sample_v2_given_v1h(self.v1_input, h)
        v2_err = tf.reduce_sum(v2_err * v2_err, [0, 1])

        return v1_err + v2_err


if __name__ == '__main__':

    n_v1 = 10
    n_v2 = 14
    n_h = 25
    n_samples = 50

    v1s = []
    v2s = []

    for n in range(n_samples):
        v1 = generate_v(n_v1, np.arange(n_v1), np.ones(n_v1))
        v1 = v1.astype(np.float32)
        v1s.append(v1)

        v2 = generate_v(n_v2, np.arange(n_v2), np.ones(n_v2))
        v2 = v2.astype(np.float32)
        v2s.append(v2)

    v1s = np.stack(v1s)
    v2s = np.stack(v2s)

    rbm = RBM(name='rbm', v1_size=n_v1, h_size=n_h, v2_size=n_v2, n_samples=n_samples, learning_rate=0.1, num_epochs=500)
    rbm.train(v1s, v2s)


