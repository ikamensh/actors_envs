from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Activation
import tensorflow as tf
import numpy as np
import time
import os

def create_q_network(tensor_in, n_actions):

    inp = Input(tensor=tensor_in)

    x = Dense(32, activation='selu', kernel_regularizer='l1') (inp)
    x = Dense(32, activation='selu', kernel_regularizer='l1')(x)
    x = Dense(32, activation='selu', kernel_regularizer='l1')(x)

    q = Dense(n_actions) (x)

    p = Activation(activation='softmax') (q)
    explaratory_choice = tf.multinomial(p, 1)

    q_net = Model(inp, q)

    return q_net, explaratory_choice


class DQN:
    def __init__(self, input_shape, n_actions, disc_rate=0.98):
        input_shape = [i for i in input_shape]
        if 'tuple' in str(type(n_actions)):
            n_actions = [i for i in input_shape]
        # q values for taking actions
        self.state =  tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
        q, expl_choice = create_q_network(self.state, n_actions)
        self.q_net = q
        self.q_eval = q.output
        self.expl_choice = expl_choice
        self.sess = tf.Session()


        # graph for updating q using experiences in format (S, a, R, S')
        self.s = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
        self.sn = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        a_ind = self.a[:,0]
        rng = tf.reshape(tf.range(tf.shape(a_ind)[0]), shape=[-1])
        indices = tf.stack([rng, a_ind], axis=1)
        pred_q = tf.gather_nd(q(self.s), indices)  # indices are numbers of rows + action index in this row

        target = self.r + disc_rate * tf.reduce_max(q(self.sn))  # target according to TD0 formula

        loss_mse = tf.reduce_mean(tf.square(pred_q - target))

        dirname = "summaries/"+str(time.time())
        os.mkdir(dirname)
        self.writter = tf.summary.FileWriter(dirname, graph= self.sess.graph, flush_secs=30)


        tf.summary.scalar("mse loss", loss_mse)
        self.summaries = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1 = 0.5, beta2=0.9)
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
        self.training_step = optimizer.minimize(loss_mse, global_step=self.global_step)

        #last step: init all tensorflow variables
        self.sess.run(tf.global_variables_initializer())

    def evaluate(self, state):
        return self.sess.run(self.q_eval, feed_dict={self.state : state})

    def greedy_choice(self, obs):
        obs = np.reshape(obs, [1, -1])
        q = self.sess.run(self.q_eval, feed_dict={self.state : obs})
        return np.argmax(q, axis=1)[0]

    def exploratory_choice(self, obs):
        obs = np.reshape(obs, [1,-1])
        return self.sess.run(self.expl_choice, feed_dict={self.state: obs})[0][0]

    def train_on_exp(self, exp):
        s, a, r, sn = exp.values
        summary, step, _ = self.sess.run([self.summaries, self.global_step, self.training_step], feed_dict={
            self.s:s, self.a:a, self.r:r,
                                                                          self.sn:sn})
        self.writter.add_summary(summary, step)



