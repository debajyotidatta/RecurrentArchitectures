import tensorflow as tf
rnn_cell = tf.nn.rnn_cell
# from tensorflow.models.rnn import rnn_cell

class GruCell(rnn_cell.RNNCell):
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
            W_r = get_variable("W_i", [self.input_size, self._num_blocks])
            W_h = get_variable("W_f", [self.input_size, self._num_blocks])


            R_z = get_variable("R_z", [self._num_blocks, self._num_blocks])
            R_r = get_variable("R_i", [self._num_blocks, self._num_blocks])
            R_h = get_variable("R_f", [self._num_blocks, self._num_blocks])

            b_z = get_variable("b_z", [1, self._num_blocks])
            b_r = get_variable("b_i", [1, self._num_blocks])
            b_h = get_variable("b_f", [1, self._num_blocks])

            g = h = tf.tanh

            r = tf.sigmoid(tf.matmul(inputs, W_r) + tf.matmul(y_prev, R_r) + b_r)
            z = tf.sigmoid(tf.matmul(inputs, W_z) + tf.matmul(y_prev, R_z) + b_z)
            h_t = g(tf.matmul(inputs, W_h) + tf.matmul(tf.mul(r, y_prev), R_h)+ b_h)
            y = tf.mul(z, y_prev) + tf.mul((1-z), h_t)

            return y, tf.concat(1, [y, y])
