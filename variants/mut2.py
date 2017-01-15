import tensorflow as tf
rnn_cell = tf.nn.rnn_cell
# from tensorflow.models.rnn import rnn_cell

class MUT2(rnn_cell.RNNCell):
    def __init__(self, num_blocks):
        self._num_blocks = num_blocks

    @property
    def input_size(self):
        return self._num_blocks

    @property
    def output_size(self):
        return self._num_blocks

    @property
    def state_size(self):
        return 2 * self._num_blocks

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)

            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

            y_prev, y_prev = tf.split(1, 2, state)
            print("y_prev shape is ", y_prev.get_shape())

            W_z = get_variable("W_z", [self.input_size, self._num_blocks])
            W_h = get_variable("W_h", [self.input_size, self._num_blocks])
            # W_f = get_variable("W_f", [self.input_size, self._num_blocks])
            # W_o = get_variable("W_o", [self.input_size, self._num_blocks])

            R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
            R_r = get_variable("R_r", [self._num_blocks, self._num_blocks])
            R_h = get_variable("R_h", [self._num_blocks, self._num_blocks])
            # R_o = get_variable("R_o", [self._num_blocks, self._num_blocks])

            b_z = get_variable("b_z", [1, self._num_blocks])
            b_r = get_variable("b_r", [1, self._num_blocks])
            b_h = get_variable("b_h", [1, self._num_blocks])
            # b_o = get_variable("b_o", [1, self._num_blocks])
            #
            # p_i = get_variable("p_i", [self._num_blocks])
            # p_f = get_variable("p_f", [self._num_blocks])
            # p_o = get_variable("p_o", [self._num_blocks])

            g = h = tf.tanh

            z = tf.sigmoid(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
            r = tf.sigmoid(inputs + tf.matmul(y_prev, R_r) + b_r)
            y = tf.mul(g(tf.matmul(tf.mul(r, y_prev), R_h) + tf.matmul(inputs, W_h) + b_h), z) + tf.mul(y_prev, (1-z))

            # z = g(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
            # i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(y_prev, R_i) + tf.mul(c_prev, p_i) + b_i)
            # f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(y_prev, R_f) + tf.mul(c_prev, p_f) + b_f)
            # c = tf.mul(i, z) + tf.mul(f, c_prev)
            # o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(y_prev, R_o) + tf.mul(c, p_o) + b_o)
            # y = tf.mul(h(c), o)

            return y, tf.concat(1, [y, y])
