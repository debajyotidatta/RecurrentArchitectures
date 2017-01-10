import tensorflow as tf
rnn_cell = tf.nn.rnn_cell

class VanillaRNNCell(rnn_cell.RNNCell):
    def __init__(self,num_blocks):
        self._num_blocks = num_blocks


    @property
    def input_size(self):
        return self._num_blocks

    @property
    def output_size(self):
        return self._num_blocks

    @property
    def state_size(self):
        return 2*self._num_blocks


    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)

            def get_variable(name, shape):
                return tf.get_variable(name, shape, initializer=initializer, dtype=inputs.dtype)

            c_prev, y_prev = tf.split(1, 2, state)

            U = get_variable("U", [self.input_size, self._num_blocks])

            W = get_variable("W", [self._num_blocks, self._num_blocks])
            V = get_variable("V", [self._num_blocks, self._num_blocks])


            b_1 = get_variable("b_1", [1, self._num_blocks])
            b_2 = get_variable("b_2", [1, self._num_blocks])


            g = h = tf.tanh
            s_t = g(tf.matmul(inputs, U) + tf.matmul(y_prev, W) + b_1)

            y = tf.sigmoid(tf.matmul(s_t, V) + b_2)

            return y, tf.concat(1, [y, y])
