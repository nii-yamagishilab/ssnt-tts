# ==============================================================================
# Copyright (c) 2018-2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
from tensorflow.python.keras import backend
import numpy as np
import math
from collections import namedtuple
from functools import reduce


class Direction:
    FORWARD = "FORWARD"
    REVERSE = "REVERSE"


class InvertibleDiagonalNorm(tf.layers.Layer):
    def __init__(self, units, epsilon=None,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(InvertibleDiagonalNorm, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.units = units
        self._epsilon = epsilon or backend.epsilon()

    def call(self, inputs, direction=Direction.FORWARD):
        x, condition = inputs
        assert condition.shape[-1].value == self.units * 2
        assert x.shape[-1].value == self.units
        if direction == Direction.FORWARD:
            return self._forward(x, condition)
        elif direction == Direction.REVERSE:
            return self._reverse(x, condition)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _forward(self, x, condition):
        x = tf.expand_dims(x, axis=1) if x.shape.rank == 2 else x
        shift, log_scale = tf.split(condition, [self.units, self.units], axis=-1)
        scale = tf.exp(log_scale) + self._epsilon
        # (B, T)
        logdet = -tf.reduce_sum(tf.log(scale), axis=-1, keep_dims=condition.shape.rank == 2)
        return (x - shift) / scale, logdet

    def _reverse(self, z, condition):
        shift, log_scale = tf.split(condition, [self.units, self.units], axis=-1)
        scale = tf.exp(log_scale) + self._epsilon
        # (B, T)
        logdet = tf.reduce_sum(tf.log(scale), axis=-1, keep_dims=condition.shape.rank == 2)
        return z * scale + shift, logdet


class InvertibleMeanOnlyNorm(tf.layers.Layer):
    def __init__(self, units,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(InvertibleMeanOnlyNorm, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.units = units

        self._linear = tf.layers.Dense(units, dtype=dtype)

    def call(self, inputs, direction=Direction.FORWARD):
        x, condition = inputs
        if direction == Direction.FORWARD:
            return self._forward(x, condition)
        elif direction == Direction.REVERSE:
            return self._reverse(x, condition)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _forward(self, x, condition):
        x = tf.expand_dims(x, axis=1) if x.shape.rank == 2 and condition.shape.rank == 3 else x
        condition = tf.expand_dims(condition, axis=1) if x.shape.rank == 3 and condition.shape.rank == 2 else condition
        shift = self._linear(condition)
        logdet = 0.0
        return x - shift, logdet

    def _reverse(self, z, condition):
        z = tf.expand_dims(z, axis=1) if z.shape.rank == 2 and condition.shape.rank == 3 else z
        condition = tf.expand_dims(condition, axis=1) if z.shape.rank == 3 and condition.shape.rank == 2 else condition
        shift = self._linear(condition)
        logdet = 0.0
        return z + shift, logdet


class AutoregressiveMask:
    MASK_INCLUSIVE = "inclusive"
    MASK_EXCLUSIVE = "exclusive"

    def __init__(self, num_blocks, n_in, n_out, mask_type=MASK_EXCLUSIVE):
        self.num_blocks = num_blocks
        self.n_in = n_in
        self.n_out = n_out
        self.mask_type = mask_type

    def _gen_slices(self):
        """Generate the slices for building an autoregressive mask."""
        slices = []
        col = 0
        d_in = self.n_in // self.num_blocks
        d_out = self.n_out // self.num_blocks
        row = d_out if self.mask_type == AutoregressiveMask.MASK_EXCLUSIVE else 0
        for _ in range(self.num_blocks):
            row_slice = slice(row, None)
            col_slice = slice(col, col + d_in)
            slices.append([row_slice, col_slice])
            col += d_in
            row += d_out
        return slices

    def gen_mask(self, dtype=tf.float32):
        """Generate the mask for building an autoregressive dense layer."""
        mask = np.zeros([self.n_out, self.n_in], dtype=dtype.as_numpy_dtype())
        slices = self._gen_slices()
        for [row_slice, col_slice] in slices:
            mask[row_slice, col_slice] = 1
        return mask.T

    def reverse(self):
        return AutoregressiveReverseMask(self.num_blocks,
                                         self.n_in,
                                         self.n_out,
                                         self.mask_type)


class AutoregressiveReverseMask:
    MASK_INCLUSIVE = "inclusive"
    MASK_EXCLUSIVE = "exclusive"

    def __init__(self, num_blocks, n_in, n_out, mask_type=MASK_EXCLUSIVE):
        self.num_blocks = num_blocks
        self.n_in = n_in
        self.n_out = n_out
        self.mask_type = mask_type

    def _gen_slices(self):
        """Generate the slices for building an autoregressive mask."""
        slices = []
        d_in = self.n_in // self.num_blocks
        d_out = self.n_out // self.num_blocks
        col = self.n_out - d_out if self.mask_type == AutoregressiveMask.MASK_EXCLUSIVE else self.n_out
        row = self.n_in
        for _ in range(self.num_blocks):
            row_slice = slice(row - d_in, row)
            col_slice = slice(None, col)
            slices.append([row_slice, col_slice])
            col -= d_out
            row -= d_in
        return slices

    def gen_mask(self, dtype=tf.float32):
        """Generate the mask for building an autoregressive dense layer."""
        mask = np.zeros([self.n_in, self.n_out], dtype=dtype.as_numpy_dtype())
        slices = self._gen_slices()
        for [row_slice, col_slice] in slices:
            mask[row_slice, col_slice] = 1
        return mask

    def reverse(self):
        return AutoregressiveMask(self.num_blocks,
                                  self.n_in,
                                  self.n_out,
                                  self.mask_type)


class InputOrder:
    ASCENT = 'ascent'
    DESCENT = 'descent'


class MaskedDense(tf.layers.Layer):

    def __init__(self, units,
                 num_blocks,
                 input_depth,
                 mask_type,
                 input_order=InputOrder.ASCENT,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=tf.initializers.glorot_normal(),
                 bias_initializer=tf.initializers.zeros(),
                 trainable=True, name=None, dtype=None, **kwargs):
        super(MaskedDense, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._units = units
        self._num_blocks = num_blocks
        self._input_depth = input_depth
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

        mask = AutoregressiveMask(self._num_blocks, input_depth, self._units, mask_type=mask_type)
        mask = mask.reverse().gen_mask() if input_order == InputOrder.DESCENT else mask.gen_mask()

        def masked_kernel_initializer(shape, dtype=None, partition_info=None):
            return mask * self._kernel_initializer(shape, dtype, partition_info)

        def masked_kernel_constraint(x):
            return mask * x

        self._dense = tf.layers.Dense(self._units,
                                      activation=self._activation,
                                      use_bias=self._use_bias,
                                      kernel_initializer=masked_kernel_initializer,
                                      kernel_constraint=masked_kernel_constraint,
                                      bias_initializer=self._bias_initializer,
                                      dtype=self.dtype)

    def build(self, input_shape):
        input_depth = input_shape.with_rank_at_least(1)[-1].value
        assert input_depth == self._input_depth, f"{input_depth} != {self._input_depth}"
        self.built = True

    def call(self, inputs, **kwargs):
        return self._dense(inputs)


class InvertibleMADE(tf.layers.Layer):

    def __init__(self, out_units,
                 num_blocks,
                 num_units,
                 num_hidden_layers,
                 input_order=InputOrder.ASCENT,
                 activation=tf.nn.relu,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(InvertibleMADE, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        assert num_hidden_layers > 0
        assert out_units % num_blocks == 0
        self._out_units = out_units
        self._num_units = num_units
        self._num_hidden_layers = num_hidden_layers
        self._input_order = input_order
        self._activation = activation

        self._condition_projection = tf.layers.Dense(num_units, use_bias=False, dtype=dtype,
                                                     name="condition_projection")

        self._hidden_layers = [MaskedDense(num_units,
                                           num_blocks=num_blocks,
                                           mask_type=AutoregressiveMask.MASK_EXCLUSIVE,
                                           input_order=input_order,
                                           input_depth=out_units,
                                           activation=None,
                                           dtype=dtype)] + [MaskedDense(num_units,
                                                                        num_blocks=num_blocks,
                                                                        mask_type=AutoregressiveMask.MASK_INCLUSIVE,
                                                                        input_order=input_order,
                                                                        input_depth=num_units,
                                                                        activation=None,
                                                                        dtype=dtype) for _ in
                                                            range(0, num_hidden_layers - 1)]
        self._mean_layer = MaskedDense(out_units,
                                       num_blocks=num_blocks,
                                       mask_type=AutoregressiveMask.MASK_INCLUSIVE,
                                       input_order=input_order,
                                       input_depth=num_units,
                                       activation=None,
                                       dtype=dtype)

        self._logscale_layer = MaskedDense(out_units,
                                           num_blocks=num_blocks,
                                           mask_type=AutoregressiveMask.MASK_INCLUSIVE,
                                           input_order=input_order,
                                           input_depth=num_units,
                                           activation=None,
                                           kernel_initializer=tf.initializers.zeros(),
                                           dtype=dtype)

    def call(self, inputs, direction=Direction.FORWARD):
        z, condition = inputs
        input_depth = z.shape.with_rank_at_least(1)[-1].value
        assert input_depth == self._out_units
        if direction == Direction.FORWARD:
            return self._forward(z, condition)
        elif direction == Direction.REVERSE:
            return self._reverse(z, condition)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _forward(self, x, condition):
        condition = self._condition_projection(condition)
        x, condition = self._upsample_input(x, condition)

        def reduce_fn(v, i, l):
            if i == 0:
                return self._activation(condition + l(v))
            else:
                return self._activation(l(v))

        h = reduce(lambda acc, il: reduce_fn(acc, il[0], il[1]), enumerate(self._hidden_layers), x)

        mean = self._mean_layer(h)
        log_scale = self._logscale_layer(h)
        logdet = -tf.reduce_sum(log_scale, axis=-1)
        u = (x - mean) * tf.exp(-log_scale), logdet
        return u

    def _reverse(self, u, condition):
        event_size = u.shape.with_rank_at_least(1)[-1].value
        condition_shape = condition.shape
        condition = self._condition_projection(condition)
        u, condition = self._upsample_input(u, condition)

        x = tf.zeros_like(u)
        mask = tf.zeros(event_size, dtype=u.dtype)
        dimensions = reversed(range(event_size)) if self._input_order == InputOrder.DESCENT else range(event_size)
        for d in dimensions:
            def reduce_fn(v, i, l):
                if i == 0:
                    return self._activation(condition + l(v))
                else:
                    return self._activation(l(v))

            h = reduce(lambda acc, il: reduce_fn(acc, il[0], il[1]), enumerate(self._hidden_layers), x)

            mean = self._mean_layer(h)
            log_scale = self._logscale_layer(h)
            x = u * tf.exp(log_scale) + mean
            mask = mask + tf.one_hot(d, event_size, dtype=mask.dtype)
            x = x * mask
        total_logdet = tf.reduce_sum(log_scale, axis=-1)
        x = tf.squeeze(x, axis=1) if condition_shape.rank == 2 else x
        return x, total_logdet

    def _upsample_input(self, inputs, conditional_inputs):
        if inputs.shape.ndims == 3 and conditional_inputs.shape.ndims == 3:
            return inputs, conditional_inputs
        elif inputs.shape.ndims == 2 and conditional_inputs.shape.ndims == 2:
            return tf.expand_dims(inputs, axis=1), tf.expand_dims(conditional_inputs, axis=1)
        elif inputs.shape.ndims == 2 and conditional_inputs.shape.ndims == 3:
            upsampled_input = tf.tile(tf.expand_dims(inputs, axis=1), multiples=[1, tf.shape(conditional_inputs)[1], 1])
            return upsampled_input, conditional_inputs
        elif inputs.shape.ndims == 3 and conditional_inputs.shape.ndims == 2:
            return inputs, tf.expand_dims(conditional_inputs, axis=1)
        else:
            raise ValueError(
                f"Invalid rank: {inputs.shape.ndims} and {conditional_inputs.shape.ndims}.")


class InvertibleMaskedAutoregressiveFlow(tf.layers.Layer):

    def __init__(self, num_made_layers,
                 out_units,
                 num_blocks,
                 num_units,
                 num_hidden_layers,
                 activation=tf.nn.relu,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(InvertibleMaskedAutoregressiveFlow, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)

        self._num_made_layers = num_made_layers
        self._made_layers = [InvertibleMADE(out_units,
                                            num_blocks,
                                            num_units,
                                            num_hidden_layers,
                                            input_order=InputOrder.ASCENT if i == 0 else InputOrder.DESCENT,
                                            activation=activation,
                                            dtype=dtype) for i in range(num_made_layers)]

    def call(self, inputs, direction=Direction.FORWARD):
        z, condition = inputs
        if direction == Direction.FORWARD:
            return self._forward(z, condition)
        elif direction == Direction.REVERSE:
            return self._reverse(z, condition)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _forward(self, x, _condition):
        conditions = [_condition[..., i::self._num_made_layers] for i in range(self._num_made_layers)]

        def reduce_fn(made, condition, inputs, total_log_det):
            output, logdet = made((inputs, condition), direction=Direction.FORWARD)
            return output, total_log_det + logdet

        output, log_det = reduce(lambda y_total_log_det, made_cond: reduce_fn(made_cond[0], made_cond[1],
                                                                              y_total_log_det[0], y_total_log_det[1]),
                                 zip(self._made_layers, conditions), (x, 0))
        return output, log_det

    def _reverse(self, u, _condition):
        conditions = [_condition[..., i::self._num_made_layers] for i in range(self._num_made_layers)]

        def reduce_fn(made, condition, inputs, total_log_det):
            output, logdet = made((inputs, condition), direction=Direction.REVERSE)
            return output, total_log_det + logdet

        output, log_det = reduce(lambda y_total_log_det, made_cond: reduce_fn(made_cond[0], made_cond[1],
                                                                              y_total_log_det[0], y_total_log_det[1]),
                                 reversed(list(zip(self._made_layers, conditions))), (u, 0))
        return output, log_det


class InvertibleMaskedAutoregressiveFlowWithMeanNorm(tf.layers.Layer):

    def __init__(self, num_made_layers,
                 out_units,
                 num_blocks,
                 num_units,
                 num_hidden_layers,
                 activation=tf.nn.relu,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(InvertibleMaskedAutoregressiveFlowWithMeanNorm, self).__init__(trainable=trainable,
                                                                             name=name, dtype=dtype, **kwargs)

        self._num_made_layers = num_made_layers
        self._made_layers = [InvertibleMADE(out_units,
                                            num_blocks,
                                            num_units,
                                            num_hidden_layers,
                                            input_order=InputOrder.ASCENT if i == 0 else InputOrder.DESCENT,
                                            activation=activation,
                                            dtype=dtype) for i in range(num_made_layers)]

        self._mean_norm = InvertibleMeanOnlyNorm(out_units, dtype=dtype)

    def call(self, inputs, direction=Direction.FORWARD):
        z, condition = inputs
        if direction == Direction.FORWARD:
            return self._forward(z, condition)
        elif direction == Direction.REVERSE:
            return self._reverse(z, condition)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _forward(self, x, _condition):
        n_layers = self._num_made_layers + 1
        conditions = [_condition[..., i::n_layers] for i in range(n_layers)]
        conditions_for_made, condition_for_norm = conditions[:-1], conditions[-1]

        def reduce_fn(made, condition, inputs, total_log_det):
            output, logdet = made((inputs, condition), direction=Direction.FORWARD)
            return output, total_log_det + logdet

        output, log_det = reduce(lambda y_total_log_det, made_cond: reduce_fn(made_cond[0], made_cond[1],
                                                                              y_total_log_det[0], y_total_log_det[1]),
                                 zip(self._made_layers, conditions_for_made), (x, 0))

        normalized_output, zero_logdet = self._mean_norm((output, condition_for_norm), direction=Direction.FORWARD)

        return normalized_output, log_det

    def _reverse(self, u, _condition):
        n_layers = self._num_made_layers + 1
        conditions = [_condition[..., i::n_layers] for i in range(n_layers)]
        conditions_for_made, condition_for_norm = conditions[:-1], conditions[-1]

        denormalized_input, zero_logdet = self._mean_norm((u, condition_for_norm), direction=Direction.REVERSE)

        def reduce_fn(made, condition, inputs, total_log_det):
            output, logdet = made((inputs, condition), direction=Direction.REVERSE)
            return output, total_log_det + logdet

        output, log_det = reduce(lambda y_total_log_det, made_cond: reduce_fn(made_cond[0], made_cond[1],
                                                                              y_total_log_det[0], y_total_log_det[1]),
                                 reversed(list(zip(self._made_layers, conditions_for_made))), (denormalized_input, 0))

        return output, log_det


class InvertibleDiagonalCovariance:

    def __init__(self, num_units, dtype=None):
        self.linear = tf.layers.Dense(num_units * 2, dtype=dtype)

        self._diagonal_norm = InvertibleDiagonalNorm(num_units, dtype=dtype)

    def __call__(self, inputs, direction):
        z, condition = inputs
        projected_condition = self.linear(condition)
        return self._diagonal_norm((z, projected_condition), direction=direction)


class InvertibleIdentity:

    def __init__(self, num_units, dtype=None):
        self.linear = tf.layers.Dense(num_units, dtype=dtype)

    def __call__(self, inputs, direction):
        z, condition = inputs
        log_det = 0.0
        mean = self.linear(condition)
        if direction == Direction.FORWARD:
            z = tf.expand_dims(z, axis=1) if mean.shape.ndims == 3 and z.shape.ndims == 2 else z
            return z - mean, log_det
        elif direction == Direction.REVERSE:
            return z + mean, log_det
        else:
            raise ValueError(f"Unknown direction: {direction}")


class BijectionType:
    MAF = "maf"
    MAF_NORM = "maf_norm"
    GLOW = "glow"
    AFFINECOUPLING = "affine_coupling"
    DIAGONAL = "diagonal"
    IDENTITY = "identity"


class BijectorConfig(namedtuple("BijectorConfig", ["bijection_type",
                                                   "num_layers",
                                                   "maf_bijector_num_hidden_layers",
                                                   "maf_bijector_num_hidden_units",
                                                   "maf_bijector_num_blocks",
                                                   "maf_bijector_activation_function"])):
    pass


def bijection_layer_factory(bijection_type, num_units, num_layers,
                            maf_bijector_num_hidden_layers=None,
                            maf_bijector_num_hidden_units=None,
                            maf_bijector_num_blocks=None,
                            maf_bijector_activation_function=None,
                            dtype=None):
    if isinstance(maf_bijector_activation_function, str):
        if maf_bijector_activation_function == "relu":
            maf_bijector_activation_function = tf.nn.relu
        elif maf_bijector_activation_function == "tanh":
            maf_bijector_activation_function = tf.nn.tanh
        else:
            raise ValueError(f"Unknown activation function: {maf_bijector_activation_function}")

    if bijection_type == BijectionType.MAF:
        return InvertibleMaskedAutoregressiveFlow(num_made_layers=num_layers,
                                                  out_units=num_units,
                                                  num_blocks=maf_bijector_num_blocks,
                                                  num_units=maf_bijector_num_hidden_units,
                                                  num_hidden_layers=maf_bijector_num_hidden_layers,
                                                  activation=maf_bijector_activation_function,
                                                  dtype=dtype)
    elif bijection_type == BijectionType.MAF_NORM:
        return InvertibleMaskedAutoregressiveFlowWithMeanNorm(num_made_layers=num_layers,
                                                              out_units=num_units,
                                                              num_blocks=maf_bijector_num_blocks,
                                                              num_units=maf_bijector_num_hidden_units,
                                                              num_hidden_layers=maf_bijector_num_hidden_layers,
                                                              activation=maf_bijector_activation_function,
                                                              dtype=dtype)
    elif bijection_type == BijectionType.DIAGONAL:
        return InvertibleDiagonalCovariance(num_units, dtype=dtype)
    elif bijection_type == BijectionType.IDENTITY:
        return InvertibleIdentity(num_units, dtype=dtype)
    else:
        raise ValueError(f"Unknown bijection type: {bijection_type}")


class Bijector:

    def __init__(self, bijection_layer, condition_input, output_stddev):
        self.bijection_layer = bijection_layer
        self._condition_input = condition_input
        self._output_stddev = output_stddev

    def transform_target(self, x):
        return self.bijection_layer((x, self._condition_input), direction=Direction.FORWARD)

    def transform_sample(self, z):
        return self.bijection_layer((z, self._condition_input), direction=Direction.REVERSE)

    def log_probability(self, x):
        target_dim = x.shape[-1].value
        z, logdet = self.transform_target(x)
        # (B, T)
        mahalanobis_distance = tf.reduce_sum(tf.square(z), axis=-1)
        # (B, T)
        log_prob = -0.5 / self._output_stddev ** 2 * mahalanobis_distance \
                   + self._log_normalization_constant(target_dim) + logdet
        return log_prob

    def sample_and_log_probability(self, z):
        output, _ = self.transform_sample(z)
        log_prob = self.log_probability(output)
        return output, log_prob

    def select(self, attend_idx):
        condition_input = tf.gather_nd(self._condition_input, attend_idx)
        return Bijector(self.bijection_layer, condition_input, self._output_stddev)

    def select_candidates(self, prev_attend_idx, input_lengths):
        batch_size = tf.shape(prev_attend_idx)[0]
        emit_attend_idx = tf.stack([tf.range(batch_size), prev_attend_idx], axis=1)
        emit_condition_input = tf.gather_nd(self._condition_input, emit_attend_idx)

        next_attend_idx = tf.minimum(prev_attend_idx + 1,
                                     tf.cast(input_lengths - 1, dtype=prev_attend_idx.dtype))
        shift_attend_idx = tf.stack([tf.range(batch_size), next_attend_idx], axis=1)
        shift_condition_input = tf.gather_nd(self._condition_input, shift_attend_idx)

        condition_input = tf.stack([emit_condition_input, shift_condition_input], axis=1)

        return Bijector(self.bijection_layer, condition_input,
                        output_stddev=self._output_stddev)

    def _log_normalization_constant(self, target_dim):
        return -0.5 * target_dim * (
                math.log(2. * math.pi, math.e) + 2. * math.log(self._output_stddev, math.e))
