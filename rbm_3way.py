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
                  
    def prop_v1v2_h(self, v1, v2):
        """ P(h|v1,v2) """
        wt = tf.transpose(w,perm=[1,2,0])
        wtv = tf.matmul(wt,v1)
        wtvs = tf.reduce_sum(wtv,axis=2)
        wtvsv = tf.matmul(wtvs,v2)
        return tf.nn.sigmoid(wtvsv + self.h_bias)
    
    def prop_v1h_v2(self, v1, h):
        """ P(h|v1,v2) """
        wt = tf.transpose(w,perm=[2,1,0])
        wtv = tf.matmul(wt,v1)
        wtvs = tf.reduce_sum(wtv,axis=2)
        wtvsv = tf.matmul(wtvs,h)
        return wtvsv + self.v2_bias   
    
    def prop_v2h_v1(self, v2, h):
        """ P(h|v1,v2) """
        wt = tf.transpose(w,perm=[0,1,2])
        wtv = tf.matmul(wt,v2)
        wtvs = tf.reduce_sum(wtv,axis=2)
        wtvsv = tf.matmul(wtvs,h)
        return wtvsv + self.v1_bias  
    
    def sample_v1_given_v2h(self, v2, h):
        """ generate sample of v1 from v2 and h"""
        return tf.random_normal(self.prop_v2h_v1(v2, h), self.v1_var)   
    
    def sample_v2_given_v1h(self, v1, h):
        """ generate sample of v1 from v2 and h"""
        return tf.random_normal(self.prop_v1h_v2(v1, h), self.v2_var)   
    

    def sample_h_given_v1v2(self, v_sample):
        """ Generate a sample from the hidden layer """
        return sample_prob(self.prop_v1v2_h(v1,v2))


    def pcd_k(self, visibles, learning_rate=0.1):
        " One step of contrastive divergence, with Rao-Blackwellization "
        h_start = self.propup(visibles)
        v_end = self.propdown(h_start)
        h_end = self.propup(v_end)
        w_positive_grad = tf.matmul(tf.transpose(visibles), h_start) / visibles.shape[0]
        w_negative_grad = tf.matmul(tf.transpose(v_end), h_end) / visibles.shape[0]

        update_w = self.weights.assign_add(learning_rate * (w_positive_grad - w_negative_grad))

        update_vb = self.v_bias.assign_add(learning_rate * tf.reduce_mean(visibles - v_end, 0))

        update_hb = self.h_bias.assign_add(learning_rate * tf.reduce_mean(h_start - h_end, 0))

        return [update_w, update_vb, update_hb]

    def reconstruction_error(self, dataset):
        """ The reconstruction cost for the whole dataset """
        err = tf.stop_gradient(dataset - self.gibbs_vhv(dataset)[1])
        return tf.reduce_sum(err * err)