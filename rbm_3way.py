""" An rbm implementation for TensorFlow, based closely on the one in Theano """

import tensorflow as tf
import math


def sample_prob(probs):
    """Takes a tensor of probabilities (as from a sigmoidal activation)
       and samples from all the distributions"""
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(probs.get_shape())))


class RBM(object):
    """ represents a 3-way rbm """

    def __init__(self, name, v1_size, h_size, v2_size):
        with tf.name_scope("rbm_" + name):
            self.weights = tf.Variable(
                tf.truncated_normal([v1_size, h_size, v2_size],
                                    stddev=1.0 / math.sqrt(float(v1_size+v2_size))), name="weights")
            self.h_bias = tf.Variable(tf.zeros([h_size]), name="h_bias")
            self.v1_bias = tf.Variable(tf.zeros([v1_size]), name="v1_bias")
            self.v1_var = tf.constant(tf.ones([v1_size]), name="v1_var")
            self.v2_bias = tf.Variable(tf.zeros([v2_size]), name="v1_bias")
            self.v2_var = tf.constant(tf.ones([v2_size]), name="v1_var")

    def _prop_helper(self, v1, v2, perm):
        """ perm specifies matrix orientation"""
        wt = tf.transpose(self.weights, perm=perm)
        wtv = tf.matmul(wt, v1)
        wtvs = tf.reduce_sum(wtv, axis=2)
        wtvsv = tf.matmul(wtvs, v2)
        return wtvsv
                  
    def prop_v1v2_h(self, v1, v2):
        """ P(h|v1,v2) """
        return tf.nn.sigmoid(self._prop_helper(v1, v2, [1, 2, 0]) + self.h_bias)
    
    def prop_v1h_v2(self, v1, h):
        """ P(h|v1,v2) """
        return self._prop_helper(v1, h, [2, 1, 0]) + self.v2_bias
    
    def prop_v2h_v1(self, v2, h):
        """ P(h|v1,v2) """
        return self._prop_helper(v2, h, [0, 1, 2]) + self.v1_bias
    
    def sample_v1_given_v2h(self, v2, h):
        """ generate sample of v1 from v2 and h"""
        return tf.random_normal(self.prop_v2h_v1(v2, h), self.v1_var)   
    
    def sample_v2_given_v1h(self, v1, h):
        """ generate sample of v1 from v2 and h"""
        return tf.random_normal(self.prop_v1h_v2(v1, h), self.v2_var)   

    def sample_h_given_v1v2(self, v1, v2):
        """ Generate a sample from the hidden layer """
        return sample_prob(self.prop_v1v2_h(v1, v2))

    def gibbs_vhv(self, v1, v2):
        """A Gibbs step from two visible layers"""
        h_sample = self.sample_h_given_v1v2(v1, v2)
        v1_sample = self.sample_v1_given_v2h(v2, h_sample)
        v2_sample = self.sample_v2_given_v1h(v1, h_sample)
        return [h_sample, v1_sample, v2_sample]

    def pcd_k(self, v1, v2, learning_rate=0.1, paralell=None):
        """ One step of contrastive divergence, with Rao-Blackwellization """
        h_start = self.prop_v1v2_h(v1, v2)
        v1_end = self.sample_v1_given_v2h(v2, h_start)
        v2_end = self.sample_v2_given_v1h(v1, h_start)
        h_end = self.prop_v1v2_h(v1_end, v2_end)

        # TODO: need to create a three way tensor from v1_end, v2_end, h_start
        w_positive_grad = tf.matmul(tf.transpose(visibles), h_start) / visibles.shape[0]
        w_negative_grad = tf.matmul(tf.transpose(v_end), h_end) / visibles.shape[0]

        update_w = self.weights.assign_add(learning_rate * (w_positive_grad - w_negative_grad))

        update_v1b = self.v1_bias.assign_add(learning_rate * tf.reduce_mean(v1 - v1_end, 0))

        update_v2b = self.v2_bias.assign_add(learning_rate * tf.reduce_mean(v2 - v2_end, 0))

        update_hb = self.h_bias.assign_add(learning_rate * tf.reduce_mean(h_start - h_end, 0))

        # TODO: implementation of PCD
        if paralell is not None:
            pass

        return [update_w, update_v1b, update_v2b, update_hb]

    def reconstruction_error(self, dataset):
        """ The reconstruction cost for the whole dataset """
        err = tf.stop_gradient(dataset - self.gibbs_vhv(dataset)[1:])
        return tf.reduce_sum(err * err)
