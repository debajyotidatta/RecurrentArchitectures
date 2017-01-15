import tensorflow as tf
rnn_cell = tf.nn.rnn_cell

class MUT1(rnn_cell.RNNCell):
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

            W_z = get_variable("W_z", [self.input_size, self._num_blocks])
            W_r = get_variable("W_r", [self.input_size, self._num_blocks])

            R_h = get_variable("R_h", [self._num_blocks, self._num_blocks])
            R_r = get_variable("R_r", [self._num_blocks, self._num_blocks])

            b_z = get_variable("b_z", [1, self._num_blocks])
            b_r = get_variable("b_r", [1, self._num_blocks])
            b_h = get_variable("b_h", [1, self._num_blocks])

            g = h = tf.tanh

            z = tf.sigmoid(tf.matmul(inputs, W_z) + b_z)
            r = tf.sigmoid(tf.matmul(inputs, W_r) + tf.matmul(y_prev, R_r)+ b_r)
            k = 1 - z
            y = tf.mul(g(tf.matmul(tf.mul(r,y_prev),R_h) + g(inputs) + b_h), z) + tf.mul(y_prev,k)

            return y, tf.concat(1, [y, y])
