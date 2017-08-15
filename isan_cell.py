import numpy as np
import tensorflow as tf
from scipy.linalg import orth
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell


class IsanCell_2(RNNCell):
    def __init__(self, num_units, reuse=None):
        super(IsanCell_2, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units, self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, step_inputs, state, scope=None, initialization='orthogonal'):
        n = step_inputs.shape[1].value
        d = self._num_units
        wx_ndd = tf.get_variable('Wx', shape=[n, d])
        bx_nd = tf.get_variable('bx', shape=[n, d])

        new_state = tf.reshape(tf.matmul(state, wx_ndd) + bx_nd, [-1, 1, d])
        return new_state, new_state


class IsanCell(RNNCell):
    def __init__(self, num_units, reuse=None):
        super(IsanCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, step_inputs, state, scope=None, initialization='gaussian'):
        """
        Make one step of ISAN transition.

        Args:
          step_inputs: one-hot encoded inputs, shape bs x n
          state: previous hidden state, shape bs x d
          scope: current scope
          initialization: how to initialize the transition matrices:
            orthogonal: usually speeds up training, orthogonalize Gaussian matrices
            gaussian: sample gaussian matrices with a sensible scale
        """
        d = self._num_units
        n = step_inputs.shape[1].value

        if initialization == 'orthogonal':
            wx_ndd_init = np.zeros((n, d * d), dtype=np.float32)
            for i in range(n):
                wx_ndd_init[i, :] = orth(np.random.randn(d, d)).astype(np.float32).ravel()
            wx_ndd_initializer = tf.constant_initializer(wx_ndd_init)
        elif initialization == 'gaussian':
            wx_ndd_initializer = tf.random_normal_initializer(stddev=1.0 / np.sqrt(d))
        else:
            raise Exception('Unknown init type: %s' % initialization)

        wx_ndd = tf.get_variable('Wx', shape=[n, d * d],
                                 initializer=wx_ndd_initializer)
        bx_nd = tf.get_variable('bx', shape=[n, d],
                                initializer=tf.zeros_initializer())

        # Multiplication with a 1-hot is just row selection.
        # As of Jan '17 this is faster than doing gather.
        Wx_bdd = tf.reshape(tf.matmul(step_inputs, wx_ndd), [-1, d, d])
        bx_bd = tf.reshape(tf.matmul(step_inputs, bx_nd), [-1, 1, d])

        # Reshape the state so that matmul multiplies different matrices
        # for each batch element.
        single_state = tf.reshape(state, [-1, 1, d])
        new_state = tf.reshape(tf.matmul(single_state, Wx_bdd) + bx_bd, [-1, d])
        return new_state, new_state


class ISAN(object):
    def __init__(self, n_tokens, hidden_dim, target_dim):
        self.n_tokens = n_tokens
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim

    def fprop(self, inputs):
        with tf.variable_scope('model', values=[inputs]):
            one_hot_inputs = tf.one_hot(inputs, self.n_tokens, axis=-1)
            with tf.variable_scope('rnn', values=[inputs]):
                states, _ = dynamic_rnn(cell=IsanCell(self.hidden_dim), inputs=one_hot_inputs, dtype=tf.float32)
            Wo = tf.get_variable('Wo', shape=[self.hidden_dim, self.target_dim],
                                 initializer=tf.random_normal_initializer(
                                     stddev=1.0 / (self.hidden_dim + self.target_dim) ** 2))
            bo = tf.get_variable('bo', shape=[1, self.target_dim],
                                 initializer=tf.zeros_initializer())

            bs, t = inputs.get_shape().as_list()
            logits = tf.matmul(tf.reshape(states, [t * bs, self.hidden_dim]), Wo) + bo
            logits = tf.reshape(logits, [bs, t, self.target_dim])
        return logits
