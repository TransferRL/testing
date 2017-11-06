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
            self.v1_size = v1_size
            self.v2_size = v2_size
            self.h_size = h_size
            self.weights = tf.Variable(
                tf.truncated_normal([v1_size, h_size, v2_size],
                    stddev=1.0 / math.sqrt(float((v1_size+v2_size)/2))), name="weights")
            self.h_bias = tf.Variable(tf.zeros([h_size]), name="h_bias")
            self.v1_bias = tf.Variable(tf.zeros([v1_size]), name="v1_bias")
            self.v1_var = tf.constant(tf.ones([v1_size]), name="v1_var")
            self.v2_bias = tf.Variable(tf.zeros([v2_size]), name="v1_bias")
            self.v2_var = tf.constant(tf.ones([v2_size]), name="v1_var")
            
            self.chain_h = None
            self.chain_v1 = None
            self.chain_v2 = None
                  
    def prop_v1v2_h(self, v1, v2):
        """ P(h|v1,v2) 
                Dimensions of v1 and v2: (n_samples, n_features)
        """
        wt = tf.transpose(w,perm=[1,2,0])
        wtv = tf.matmul(wt,tf.transpose(v1))
        wtvt = tf.transpose(wtv,perm=[2,0,1])
        wtvtv = tf.matmul(wtvt,tf.expand_dims(v2,axis=-1))
        wtvtvs = tf.reduce_sum(wtvtv,axis=-1)
        return tf.nn.sigmoid(wtvtvs + tf.tile(self.h_bias,(v1.shape.as_list()[0],1)))
    
    def prop_v1h_v2(self, v1, h):
        """ P(h|v1,v2) 
                Dimensions of v1 and h: (n_samples, n_features)
        """
        wt = tf.transpose(w,perm=[2,1,0])
        wtv = tf.matmul(wt,tf.transpose(v1))
        wtvt = tf.transpose(wtv,perm=[2,0,1])
        wtvtv = tf.matmul(wtvt,tf.expand_dims(h,axis=-1))
        wtvtvs = tf.reduce_sum(wtvtv,axis=-1)
        return wtvsv + tf.tile(self.v2_bias,(v1.shape.as_list()[0],1))
    
    def prop_v2h_v1(self, v2, h):
        """ P(h|v1,v2) 
                Dimensions of v2 and h: (n_samples, n_features)
        """
        wt = tf.transpose(w,perm=[0,1,2])
        wtv = tf.matmul(wt,tf.transpose(v2))
        wtvt = tf.transpose(wtv,perm=[2,0,1])
        wtvtv = tf.matmul(wtvt,tf.expand_dims(h,axis=-1))
        wtvtvs = tf.reduce_sum(wtvtv,axis=-1)
        return wtvsv + tf.tile(self.v1_bias,(v2.shape.as_list()[0],1))
    
    def sample_v1_given_v2h(self, v2, h):
        """ generate sample of v1 from v2 and h"""
        dist = tf.contrib.distributions.Normal(self.prop_v2h_v1(v2, h),[self.v1_var]*self.v1_size)
        return tf.transpose(dist.sample(1))
    
    def sample_v2_given_v1h(self, v1, h):
        """ generate sample of v1 from v2 and h"""
        dist = tf.contrib.distributions.Normal(self.prop_v1h_v2(v1, h),[self.v2_var]*self.v2_size)
        return tf.transpose(dist.sample(1))  

    def sample_h_given_v1v2(self, v1, v2):
        """ Generate a sample from the hidden layer """
        return sample_prob(self.prop_v1v2_h(v1,v2))
    
    def get_delta_products(v1,h,v2):
        v1 = tf.expand_dims(v1,axis=-1)
        h = tf.expand_dims(h,axis=-1)
        v2 = tf.expand_dims(v2,axis=-1)
        prod = tf.matmul(v1,np.transpose(h, perm=[0,2,1]))
        prod = tf.transpose(tf.expand_dims(prod,-1), perm=[1,0,2,3])
        prod = tf.matmul(prod,tf.transpose(v2, perm=[0,2,1]))
        prod = tf.transpose(prod,perm=[1,0,2,3])
        return tf.reduce_mean(prod,axis=0)
    
    def gibbs(v1,h,v2):
        
        # using activations
        v1 = self.prop_v2h_v1(v2, h)
        v2 = self.prop_v1h_v2(v1, h)
        h = self.prop_v1v2_h(v1, v2)
        
        # using sampling
        #v1 = self.sample_v1_given_v2h(v2, h)
        #v2 = self.sample_v2_given_v1h(v1, h)
        #h = sample_h_given_v1v2(v1, v2)
        
        return v1,h,v2

    def pcd_k(self, start_v1, start_v2, learning_rate=0.1, k=1, persistant=True):
        "k-step (persistent) contrastive divergence"
        if self.chain_v1 is None and persistant == True:
            self.chain_v1 = v1
        if self.chain_v2 is None and persistant == True:
            self.chain_v2 = v2
        if self.chain_h is None and persistant == True:
            self.chain_h = self.prop_v1v2_h(self.chain_v1, self.chain_v2)
        
        if persistant == True: mcmc_v1 = self.chain_v1
        else: mcmc_v1 = start_v1
        if persistant == True: mcmc_v2 = self.chain_v2
        else: mcmc_v2 = start_v2
        if persistant == True: mcmc_h = self.chain_h
        else: mcmc_h = self.prop_v1v2_h(start_v1, start_v2)
        start_h = mcmc_h.copy()
        
        for n in range(k):
            self.chain_v1, self.chain_h, self.chain_v2 = gibbs(self.chain_v1, self.chain_h, self.chain_v2)
        
        w_positive_grad = self.get_delta_products(start_v1,start_h,start_v2)
        w_negative_grad = self.get_delta_products(self.chain_v1,self.chain_h,self.chain_v2)

        update_w = self.weights.assign_add(learning_rate * (w_positive_grad - w_negative_grad))

        update_v1b = self.v1_bias.assign_add(learning_rate * tf.reduce_mean(v1_start - v1_end, 0))
        update_v2b = self.v1_bias.assign_add(learning_rate * tf.reduce_mean(v2_start - v2_end, 0))

        update_hb = self.h_bias.assign_add(learning_rate * tf.reduce_mean(h_start - h_end, 0))

        return [update_w, update_v1b, update_hb, update_v2b]

    def reconstruction_error(self, v1, v2):
        """ The one-step reconstruction cost for both visible layers """
        h = prop_v1v2_h(v1, v2)
        
        v1_err = tf.stop_gradient(v1 - sample_v1_given_v2h(self, v2, h))
        v1_err = tf.reduce_sum(v1_err * v1_err,[0,1])
        
        v2_err = tf.stop_gradient(v2 - sample_v2_given_v1h(self, v1, h))
        v2_err = tf.reduce_sum(v2_err * v2_err,[0,1])
        return v1_err + v2_err, v1_err, v2_err