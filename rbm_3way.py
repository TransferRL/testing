""" An rbm implementation for TensorFlow, based closely on the one in Theano """

import tensorflow as tf
import math
import numpy as np

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
            
            self.chain_h = None
            self.chain_v1 = None
            self.chain_v2 = None
                  
    def _prop_helper(self, a, b, perm):
        """ perm specifies matrix orientation"""
        wt = tf.transpose(self.weights,perm=perm)
        wtv = tf.einsum('ijk,kl->ijl',wt,tf.transpose(a))
        wtvt = tf.transpose(wtv,perm=[2,0,1])
        wtvtv = tf.matmul(wtvt,tf.expand_dims(b,axis=-1))
        wtvtvs = tf.reduce_sum(wtvtv,axis=-1)
        return wtvtvs
                  
    def prop_v1v2_h(self, v1, v2, n_samples):
        """ P(h|v1,v2) """
        return tf.nn.sigmoid(self._prop_helper(v1, v2, [1, 2, 0]) + tf.tile(self.h_bias,[n_samples,1]))
    
    def prop_v1h_v2(self, v1, h, n_samples):
        """ P(v2|v1,h) """
        return self._prop_helper(v1, h, [2, 1, 0]) + tf.tile(self.v2_bias,[n_samples,1])
    
    def prop_v2h_v1(self, v2, h, n_samples):
        """ P(v1|v2,h) """
        return self._prop_helper(v2, h, [0, 1, 2]) + tf.tile(self.v1_bias,[n_samples,1])
    
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
    
    def get_delta_products(self,v1,h,v2):
        v1 = tf.expand_dims(v1,axis=-1)
        h = tf.expand_dims(h,axis=-1)
        v2 = tf.expand_dims(v2,axis=-1)
        prod = tf.matmul(v1,tf.transpose(h, perm=[0,2,1]))
        prod = tf.transpose(tf.expand_dims(prod,-1), perm=[1,0,2,3])
        prod = tf.einsum('ijkl,jmn->ijkn',prod,tf.transpose(v2,perm=[0,2,1]))
        prod = tf.transpose(prod,perm=[1,0,2,3])
        return tf.reduce_mean(prod,axis=0)
    
    def gibbs(self,v1,h,v2, n_samples):
        
        # using activations
        v1 = self.prop_v2h_v1(v2, h, n_samples)
        v2 = self.prop_v1h_v2(v1, h, n_samples)
        h = self.prop_v1v2_h(v1, v2, n_samples)
        
        # using sampling
        #v1 = self.sample_v1_given_v2h(v2, h)
        #v2 = self.sample_v2_given_v1h(v1, h)
        #h = sample_h_given_v1v2(v1, v2)
        
        return v1,h,v2

    def pcd_k(self, start_v1, start_v2, learning_rate=0.1, k=1, persistant=False):
        "k-step (persistent) contrastive divergence"
        
        n_samples = len(start_v1)
        
        if self.chain_v1 is None and persistant == True:
            self.chain_v1 = start_v1
        if self.chain_v2 is None and persistant == True:
            self.chain_v2 = start_v2
        if self.chain_h is None and persistant == True:
            self.chain_h = self.prop_v1v2_h(self.chain_v1, self.chain_v2, n_samples)
        
        if persistant == True: mcmc_v1 = self.chain_v1
        else: mcmc_v1 = start_v1
        if persistant == True: mcmc_v2 = self.chain_v2
        else: mcmc_v2 = start_v2
        if persistant == True: start_h = self.chain_h
        else: start_h = self.prop_v1v2_h(start_v1, start_v2, n_samples)
        mcmc_h = start_h
        
        for n in range(k):
            mcmc_v1, mcmc_h, mcmc_v2 = self.gibbs(mcmc_v1, mcmc_h, mcmc_v2, n_samples)
            
        if persistant == True:
            self.chain_v1, self.chain_h, self.chain_v2 = mcmc_v1, mcmc_h, mcmc_v2
        
        w_positive_grad = self.get_delta_products(start_v1,start_h,start_v2)
        w_negative_grad = self.get_delta_products(mcmc_v1,mcmc_h,mcmc_v2)

        update_w = self.weights.assign_add(learning_rate * (w_positive_grad - w_negative_grad))

        update_v1b = self.v1_bias.assign_add(learning_rate * tf.reduce_mean(start_v1 - mcmc_v1, 0, keep_dims=True))
        update_v2b = self.v2_bias.assign_add(learning_rate * tf.reduce_mean(start_v2 - mcmc_v2, 0, keep_dims=True))

        update_hb = self.h_bias.assign_add(learning_rate * tf.reduce_mean(start_h - mcmc_h, 0, keep_dims=True))

        return [update_w, update_v1b, update_hb, update_v2b]

    def reconstruction_error(self, v1, v2):
        """ The one-step reconstruction cost for both visible layers """
        h = self.prop_v1v2_h(v1, v2, v1.shape[0])
        
        v1_err = tf.cast(v1,tf.float32) - self.sample_v1_given_v2h(v2, h)
        v1_err = tf.reduce_sum(v1_err * v1_err,[0,1])
        
        v2_err = tf.cast(v2,tf.float32) - self.sample_v2_given_v1h(v1, h)
        v2_err = tf.reduce_sum(v2_err * v2_err,[0,1])
        
        return v1_err + v2_err, v1_err, v2_err