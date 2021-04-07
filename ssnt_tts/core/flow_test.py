import tensorflow as tf
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from core.flow import InvertibleMADE, InvertibleMaskedAutoregressiveFlow, InvertibleDiagonalCovariance, \
    InvertibleMaskedAutoregressiveFlowWithMeanNorm, Direction, InputOrder


@composite
def input_tensor(draw, batch_size, length, dim=integers(2, 20).filter(lambda x: x % 2 == 0),
                 elements=integers(-1, 1)):
    t = length
    c = draw(dim)
    shape = [batch_size, c] if t == 0 else [batch_size, t, c]
    btc = draw(arrays(dtype=np.float32, shape=shape, elements=elements))
    return btc, shape


@composite
def condition_tensor(draw, batch_size, length, dim=integers(3, 24).filter(lambda x: x % 3 == 0),
                     elements=integers(-1, 1)):
    t = length
    c = draw(dim)
    shape = [batch_size, c] if t == 0 else [batch_size, t, c]
    btc = draw(arrays(dtype=np.float32, shape=shape, elements=elements))
    return btc, shape


@composite
def input_args(draw, batch_size=integers(1, 3), length=integers(0, 10)):
    bs = draw(batch_size)
    length = draw(length)
    input_t, input_shape = draw(input_tensor(bs, length))
    return input_t, input_shape


@composite
def input_and_condition_args(draw, batch_size=integers(1, 3), length=integers(0, 10)):
    bs = draw(batch_size)
    length = draw(length)
    input_t, input_shape = draw(input_tensor(bs, length))
    condition_t, condition_shape = draw(condition_tensor(bs, length))
    return input_t, input_shape, condition_t, condition_shape


class FlowTest(tf.test.TestCase):

    config = tf.ConfigProto(device_count={'GPU': 0})

    @given(args=input_and_condition_args())
    @settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_masked_autoregressive_flow_inversion(self, args):
        tf.set_random_seed(12345)
        inputs, input_shape, condition, condition_shape = args
        inputs = tf.convert_to_tensor(inputs)
        condition = tf.convert_to_tensor(condition)

        maf = InvertibleMADE(input_shape[-1], input_shape[-1], 16, 2, input_order=InputOrder.ASCENT)

        forward_output, log_det = maf((inputs, condition), direction=Direction.FORWARD)
        reversed_output, reversed_log_det = maf((forward_output, condition), direction=Direction.REVERSE)
        if inputs.shape.rank != reversed_output.shape.rank:
            inputs = tf.tile(tf.expand_dims(inputs, axis=1), multiples=[1, tf.shape(reversed_output)[1], 1])
            log_det = tf.tile(log_det, multiples=[1, tf.shape(reversed_output)[1]])

        with self.cached_session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            inputs, reversed_output = sess.run([inputs, reversed_output])
            self.assertAllClose(inputs, reversed_output)
            log_det, reversed_log_det = sess.run([log_det, reversed_log_det])
            self.assertAllClose(log_det, -reversed_log_det)

    @given(args=input_and_condition_args())
    @settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_reverse_masked_autoregressive_flow_inversion(self, args):
        tf.set_random_seed(12345)
        inputs, input_shape, condition, condition_shape = args
        inputs = tf.convert_to_tensor(inputs)
        condition = tf.convert_to_tensor(condition)

        maf = InvertibleMADE(input_shape[-1], input_shape[-1], 16, 2, input_order=InputOrder.DESCENT)

        forward_output, log_det = maf((inputs, condition), direction=Direction.FORWARD)
        reversed_output, reversed_log_det = maf((forward_output, condition), direction=Direction.REVERSE)
        if inputs.shape.rank != reversed_output.shape.rank:
            inputs = tf.tile(tf.expand_dims(inputs, axis=1), multiples=[1, tf.shape(reversed_output)[1], 1])
            log_det = tf.tile(log_det, multiples=[1, tf.shape(reversed_output)[1]])

        with self.cached_session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            inputs, reversed_output = sess.run([inputs, reversed_output])
            log_det, reversed_log_det = sess.run([log_det, reversed_log_det])
            self.assertAllClose(inputs, reversed_output)
            self.assertAllClose(log_det, -reversed_log_det)

    @given(args=input_and_condition_args())
    @settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_stacked_masked_autoregressive_flow_inversion(self, args):
        tf.set_random_seed(12345)
        inputs, input_shape, condition, condition_shape = args
        inputs = tf.convert_to_tensor(inputs)
        condition = tf.convert_to_tensor(condition)

        maf = InvertibleMaskedAutoregressiveFlow(2,
                                                 input_shape[-1],
                                                 input_shape[-1], 16, 2)

        forward_output, log_det = maf((inputs, condition), direction=Direction.FORWARD)
        reversed_output, reversed_log_det = maf((forward_output, condition), direction=Direction.REVERSE)
        if inputs.shape.rank != reversed_output.shape.rank:
            inputs = tf.tile(tf.expand_dims(inputs, axis=1), multiples=[1, tf.shape(reversed_output)[1], 1])
            log_det = tf.tile(log_det, multiples=[1, tf.shape(reversed_output)[1]])

        with self.cached_session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            inputs, reversed_output = sess.run([inputs, reversed_output])
            log_det, reversed_log_det = sess.run([log_det, reversed_log_det])
            self.assertAllClose(inputs, reversed_output)
            self.assertAllClose(log_det, -reversed_log_det)


    @given(args=input_and_condition_args())
    @settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_diagonal_norm_inversion(self, args):
        tf.set_random_seed(12345)
        inputs, input_shape, condition, condition_shape = args
        inputs = tf.convert_to_tensor(inputs)
        condition = tf.convert_to_tensor(condition)

        diagonal_norm = InvertibleDiagonalCovariance(input_shape[-1])

        forward_output, log_det = diagonal_norm((inputs, condition), direction=Direction.FORWARD)
        reversed_output, reversed_log_det = diagonal_norm((forward_output, condition), direction=Direction.REVERSE)

        if inputs.shape.rank != reversed_output.shape.rank:
            inputs = tf.tile(tf.expand_dims(inputs, axis=1), multiples=[1, tf.shape(reversed_output)[1], 1])

        with self.cached_session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            inputs, reversed_output = sess.run([inputs, reversed_output])
            self.assertAllClose(inputs, reversed_output)
            log_det, reversed_log_det = sess.run([log_det, reversed_log_det])
            self.assertAllClose(log_det, -reversed_log_det)

    @given(args=input_and_condition_args())
    @settings(deadline=None, max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_stacked_masked_autoregressive_flow_with_mean_norm_inversion(self, args):
        tf.set_random_seed(12345)
        inputs, input_shape, condition, condition_shape = args
        inputs = tf.convert_to_tensor(inputs)
        condition = tf.convert_to_tensor(condition)

        maf = InvertibleMaskedAutoregressiveFlowWithMeanNorm(2,
                                                             input_shape[-1],
                                                             input_shape[-1], 16, 1)

        forward_output, log_det = maf((inputs, condition), direction=Direction.FORWARD)
        reversed_output, reversed_log_det = maf((forward_output, condition), direction=Direction.REVERSE)
        if inputs.shape.rank != reversed_output.shape.rank:
            inputs = tf.tile(tf.expand_dims(inputs, axis=1), multiples=[1, tf.shape(reversed_output)[1], 1])
            log_det = tf.tile(log_det, multiples=[1, tf.shape(reversed_output)[1]])

        with self.cached_session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            inputs, reversed_output = sess.run([inputs, reversed_output])
            log_det, reversed_log_det = sess.run([log_det, reversed_log_det])
            self.assertAllClose(inputs, reversed_output)
            self.assertAllClose(log_det, -reversed_log_det)


if __name__ == '__main__':
    tf.test.main()
