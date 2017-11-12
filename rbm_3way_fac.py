""" An rbm implementation for TensorFlow, based closely on the one in Theano """

import tensorflow as tf
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools


def sample_prob(probs):
    """Takes a tensor of probabilities (as from a sigmoidal activation)
       and samples from all the distributions"""
    return tf.nn.relu(
        tf.sign(
            probs - tf.random_uniform(probs.get_shape())))


def generate_v(n_v, mu_v, sig_v):
    v = np.random.normal(mu_v, sig_v, n_v)
    return v


def gen_batches(data, batch_size):
    """Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i + batch_size]


class RBM(object):
    """ represents a 3-way rbm """

    def __init__(self, name, v1_size, h_size, v2_size, n_data, batch_size, num_epochs=100, learning_rate=0.1, k=1,
                 use_tqdm=True, show_err_plt=True,n_factors=50):
        with tf.name_scope("rbm_" + name):
            self.v1_size = v1_size
            self.v2_size = v2_size
            self.h_size = h_size
            self.fweights_v1 = tf.Variable(
                tf.truncated_normal([v1_size, n_factors],
                                    stddev=1.0 / math.sqrt(float((v1_size + v2_size) / 2))), name="weights")
            self.fweights_v2 = tf.Variable(
                tf.truncated_normal([v2_size, n_factors],
                                    stddev=1.0 / math.sqrt(float((v1_size + v2_size) / 2))), name="weights")
            self.fweights_h = tf.Variable(
                tf.truncated_normal([h_size, n_factors],
                                    stddev=1.0 / math.sqrt(float((v1_size + v2_size) / 2))), name="weights")
            self.h_bias = tf.Variable(tf.zeros([1, h_size]), name="h_bias",dtype=tf.float32)
            self.v1_bias = tf.Variable(tf.zeros([1, v1_size]), name="v1_bias",dtype=tf.float32)
            self.v1_var = tf.constant(np.ones([v1_size]), name="v1_var",dtype=tf.float32)
            self.v2_bias = tf.Variable(tf.zeros([1, v2_size]), name="v1_bias",dtype=tf.float32)
            self.v2_var = tf.constant(np.ones([v2_size]), name="v1_var",dtype=tf.float32)

            self.batch_size = batch_size
            self.n_batches = n_data // batch_size # assume it will be an integer

            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.k = k

            self.use_tqdm = use_tqdm
            self.show_err_plt = show_err_plt

            self.v1_input = tf.placeholder('float32', (self.batch_size, self.v1_size))
            self.v2_input = tf.placeholder('float32', (self.batch_size, self.v2_size))

            self.compute_err = None  # filled in reconstruction error
            self.tf_session = None
            
            self.cost = []

            self.final_h = None

    def _prop_helper(self, a, b, a_weights, b_weights, t_weights):
        """a and b should be matricies of row vectors"""
        inter = tf.multiply(tf.matmul(a, a_weights),tf.matmul(b, b_weights))
        return tf.matmul(inter,tf.transpose(t_weights))

    def prop_v1v2_h(self, v1, v2):
        """ P(h|v1,v2) """
        return tf.nn.sigmoid(self._prop_helper(v1, v2, self.fweights_v1, self.fweights_v2, self.fweights_h) + self.h_bias)

    def prop_v1h_v2(self, v1, h):
        """ P(v2|v1,h) """
        return self._prop_helper(v1, h, self.fweights_v1, self.fweights_h, self.fweights_v2) + self.v2_bias

    def prop_v2h_v1(self, v2, h):
        """ P(v1|v2,h) """
        return self._prop_helper(v2, h, self.fweights_v2, self.fweights_h, self.fweights_v1) + self.v1_bias

    def sample_v1_given_v2h(self, v2, h):
        """ generate sample of v1 from v2 and h"""
        dist = tf.contrib.distributions.Normal(tf.cast(self.prop_v2h_v1(v2, h), tf.float32),
                                               tf.cast(tf.tile(tf.expand_dims(self.v1_var, 0), [v2.get_shape().as_list()[0], 1]),
                                                       tf.float32))
        return tf.reduce_sum(dist.sample(1), 0)

    def sample_v2_given_v1h(self, v1, h):
        """ generate sample of v1 from v2 and h"""
        dist = tf.contrib.distributions.Normal(tf.cast(self.prop_v1h_v2(v1, h), tf.float32),
                                               tf.cast(tf.tile(tf.expand_dims(self.v2_var, 0), [v1.get_shape().as_list()[0], 1]),
                                                       tf.float32))
        return tf.reduce_sum(dist.sample(1), 0)

    def sample_h_given_v1v2(self, v1, v2):
        """ Generate a sample from the hidden layer """
        return sample_prob(self.prop_v1v2_h(v1, v2))

    @staticmethod
    def get_delta_products(t, a, b, a_weights, b_weights):
        """ inputs are normalized feature vectors (i.e. v1/v1_var)"""
        inter = tf.multiply(tf.matmul(a,a_weights),tf.matmul(b,b_weights))
        return tf.matmul(tf.transpose(t),inter)

    def gibbs(self, v1, h, v2):

        # using mean field values
        v1 = self.prop_v2h_v1(v2, h)
        v2 = self.prop_v1h_v2(v1, h)
        h = self.prop_v1v2_h(v1, v2)

        # using sampling
        # v1 = self.sample_v1_given_v2h(v2, h)
        # v2 = self.sample_v2_given_v1h(v1, h)
        # h = sample_h_given_v1v2(v1, v2)

        return v1, h, v2

    def train(self, v1_input, v2_input):
        """train RBM"""

        self.pcd_k()  # define pcd step
        self.reconstruction_error()  # define error metric

        self.tf_session = tf.Session()
        init = tf.global_variables_initializer()
        self.tf_session.run(init)

        pbar = tqdm(range(self.num_epochs))
        for i in pbar:
            avg_err = self.one_train_step(v1_input, v2_input)
            self.cost.append(avg_err)
            pbar.set_description('squared reconstruction average batch error: {}'.format(avg_err))

            # catch divergence
            if np.isnan(self.cost[-1]) == True:
                raise RuntimeError('Training has diverged - lower learning rate!')
            # early stopping
            if i > 20 and np.mean(self.cost[-20:-10]) - np.mean(self.cost[-10:]) < np.mean(self.cost[-20:]) * 0.01:
                break

        return self.cost

    def one_train_step(self, v1_input, v2_input):
        """run one training step"""

        updates = [self.fweights_v1, self.fweights_v2, self.fweights_h, self.v1_bias, self.v2_bias, self.h_bias]
        err_tot = 0
        for i in range(self.n_batches):
            np.random.shuffle(v1_input)
            np.random.shuffle(v2_input)
            v1_input_list = np.split(v1_input, self.n_batches)
            v2_input_list = np.split(v2_input, self.n_batches)

            self.tf_session.run(updates, feed_dict={self.v1_input: v1_input_list[i], self.v2_input: v2_input_list[i]})
            err_tot += self.get_cost(v1_input_list[i],v2_input_list[i])
        return err_tot / (self.batch_size * self.n_batches)

    def pcd_k(self):
        "k-step (persistent) contrastive divergence"

        mcmc_v1, mcmc_v2 = (self.v1_input, self.v2_input)

        start_h = self.prop_v1v2_h(self.v1_input, self.v2_input)
        mcmc_h = start_h

        for n in range(self.k):
            mcmc_v1, mcmc_h, mcmc_v2 = self.gibbs(mcmc_v1, mcmc_h, mcmc_v2)

        self.final_h = mcmc_h

        # update fweights_v1
        fw_v1_positive_grad = self.get_delta_products(tf.divide(self.v1_input,self.v1_var), start_h, tf.divide(self.v2_input,self.v2_var),self.fweights_h,self.fweights_v2) / self.batch_size
        fw_v1_negative_grad = self.get_delta_products(tf.divide(mcmc_v1,self.v1_var), mcmc_h, tf.divide(mcmc_v2,self.v2_var),self.fweights_h,self.fweights_v2) / self.batch_size
        self.fweights_v1 = self.fweights_v1.assign_add(self.learning_rate * (fw_v1_positive_grad - fw_v1_negative_grad))

        # update fweights_v2
        fw_v2_positive_grad = self.get_delta_products(tf.divide(self.v2_input,self.v2_var), start_h, tf.divide(self.v1_input,self.v1_var),self.fweights_h,self.fweights_v1) / self.batch_size
        fw_v2_negative_grad = self.get_delta_products(tf.divide(mcmc_v2,self.v2_var), mcmc_h, tf.divide(mcmc_v1,self.v1_var),self.fweights_h,self.fweights_v1) / self.batch_size
        self.fweights_v2 = self.fweights_v2.assign_add(self.learning_rate * (fw_v2_positive_grad - fw_v2_negative_grad))

        # update fweights_h
        fw_h_positive_grad = self.get_delta_products(start_h, tf.divide(self.v2_input,self.v2_var), tf.divide(self.v1_input,self.v1_var),self.fweights_v2,self.fweights_v1) / self.batch_size
        fw_h_negative_grad = self.get_delta_products(mcmc_h, tf.divide(mcmc_v2,self.v2_var), tf.divide(mcmc_v1,self.v1_var),self.fweights_v2,self.fweights_v1) / self.batch_size
        self.fweights_h = self.fweights_h.assign_add(self.learning_rate * (fw_h_positive_grad - fw_h_negative_grad))

        self.v1_bias = self.v1_bias.assign_add(self.learning_rate * tf.reduce_mean(self.v1_input - mcmc_v1, 0,
                                                                                   keep_dims=True))
        self.v2_bias = self.v2_bias.assign_add(self.learning_rate * tf.reduce_mean(self.v2_input - mcmc_v2, 0,
                                                                                   keep_dims=True))

        self.h_bias = self.h_bias.assign_add(self.learning_rate * tf.reduce_mean(start_h - mcmc_h, 0, keep_dims=True))


    def get_cost(self, v1_input, v2_input):

        return self.tf_session.run(self.compute_err, feed_dict={self.v1_input: v1_input,
                                                                self.v2_input: v2_input})

    def reconstruction_error(self):
        """ The one-step reconstruction cost for both visible layers """
        h = self.prop_v1v2_h(self.v1_input, self.v2_input)

        v1_err = tf.cast(self.v1_input, tf.float32) - self.sample_v1_given_v2h(self.v2_input, h)
        v1_err = tf.reduce_sum(v1_err * v1_err, [0, 1])

        v2_err = tf.cast(self.v2_input, tf.float32) - self.sample_v2_given_v1h(self.v1_input, h)
        v2_err = tf.reduce_sum(v2_err * v2_err, [0, 1])

        self.compute_err = v1_err + v2_err

    def v2_predict(self,v1_input):

        # mean field
        #v2_predictions = self.prop_v1h_v2(v1_inputs, self.h)
        # sample
        v1_input_list = np.split(v1_input, self.n_batches)
        v2_predictions = []
        for i in range(self.n_batches):
            v2_prediction = self.sample_v2_given_v1h(self.v1_input, self.final_h)
            v2_predictions.append(self.tf_session.run(v2_prediction,feed_dict={self.v1_input: v1_input_list[i], self.v2_input: np.zeros([self.batch_size,self.v2_size])}))
        return np.stack(v2_predictions).reshape(-1,self.v2_size)


def main():

    n_v1 = 30
    n_v2 = 30
    n_h = 60
    n_samples = 5000

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
    v2s = np.stack(v1s)

    print(v1s.shape[0])

    rbm = RBM(name='rbm', v1_size=n_v1, h_size=n_h, v2_size=n_v2
              , n_data = v1s.shape[0], batch_size=100, learning_rate=0.0000001,
              num_epochs=500, n_factors=10)
    errs = rbm.train(v1s, v2s)
    print('getting predictions')
    print('v2s:', v2s[:3])
    v2_predictions = rbm.v2_predict(v1s)
    print('v2 preds:', v2_predictions[:3])

    print('preds diff:',(v2_predictions - v2s).mean(axis=0))

    rbm.tf_session.close()


    if rbm.show_err_plt:
        plt.plot(range(len(rbm.cost)), rbm.cost)
        plt.show()

if __name__ == '__main__':
    main()