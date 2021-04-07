# ==============================================================================
# Copyright (c) 2018-2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

from collections import namedtuple
import tensorflow as tf
from tensorflow.python.keras import backend
import math
from typing import Tuple
from abc import abstractmethod
from functools import reduce
from core.modules import PreNet, ZoneoutLSTMCell
from core.modules import DecoderPreNetWrapper
from core.modules import LSTMImpl
from tensorflow.contrib.seq2seq import BasicDecoder
from tensorflow.contrib.seq2seq import Helper
from core.flow import Bijector, bijection_layer_factory, BijectionType, BijectorConfig


def random_logistic(shape, dtype, epsilon):
    uniform_noise = tf.random_uniform(shape, minval=epsilon, maxval=1 - epsilon, dtype=dtype)
    return tf.log(uniform_noise) - tf.log(1 - uniform_noise)


class AlignmentSampler:

    @property
    @abstractmethod
    def mode(self):
        raise NotImplementedError("AlignmentSampler#mode")

    @abstractmethod
    def sample(self, emit_prob_logits, inverse_temperature, prev_attend_idx):
        raise NotImplementedError("AlignmentSampler#sample")


class AlignmentProbabilitySample(namedtuple("AlignmentProbabilitySample", ["log_emit_probs",
                                                                           "log_shift_probs",
                                                                           "prev_attend_idx"])):
    pass


class DeterministicAlignmentSampler(AlignmentSampler):
    mode = "deterministic"

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def sample(self, emit_prob_logits, inverse_temperature, prev_attend_idx):
        emit_prob_logits = emit_prob_logits * inverse_temperature
        log_emit_probs = tf.math.log_sigmoid(emit_prob_logits)
        # log(1 - sigmoid(x)) = -x + log(sigmoid(x))
        log_shift_probs = - emit_prob_logits + log_emit_probs

        return AlignmentProbabilitySample(log_emit_probs, log_shift_probs, prev_attend_idx)


class StochasticLogisticAlignmentSampler(AlignmentSampler):
    mode = "stochastic"
    distribution = "logistic"

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def sample(self, emit_prob_logits, inverse_temperature, prev_attend_idx):
        emit_prob_logits = emit_prob_logits * inverse_temperature
        log_emit_probs = tf.math.log_sigmoid(emit_prob_logits)
        # log(1 - sigmoid(x)) = -x + log(sigmoid(x))
        log_shift_probs = - emit_prob_logits + log_emit_probs

        log_emit_probs = log_emit_probs + random_logistic(tf.shape(emit_prob_logits),
                                                          dtype=emit_prob_logits.dtype,
                                                          epsilon=self._epsilon)

        return AlignmentProbabilitySample(log_emit_probs, log_shift_probs, prev_attend_idx)


class StochasticBinConcreteAlignmentSampler(AlignmentSampler):
    mode = "stochastic"
    distribution = "binconcrete"

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def sample(self, emit_prob_logits, inverse_temperature, prev_attend_idx):
        emit_prob_logits = emit_prob_logits + random_logistic(tf.shape(emit_prob_logits),
                                                              dtype=emit_prob_logits.dtype,
                                                              epsilon=self._epsilon)
        emit_prob_logits = emit_prob_logits * inverse_temperature
        log_emit_probs = tf.math.log_sigmoid(emit_prob_logits)
        # log(1 - sigmoid(x)) = -x + log(sigmoid(x))
        log_shift_probs = - emit_prob_logits + log_emit_probs

        return AlignmentProbabilitySample(log_emit_probs, log_shift_probs, prev_attend_idx)


def alignment_sampler_factory(mode, distribution, epsilon):
    if mode == DeterministicAlignmentSampler.mode:
        return DeterministicAlignmentSampler(epsilon)
    elif mode == StochasticLogisticAlignmentSampler.mode:
        if distribution == StochasticLogisticAlignmentSampler.distribution:
            return StochasticLogisticAlignmentSampler(epsilon)
        elif distribution == StochasticBinConcreteAlignmentSampler.distribution:
            return StochasticBinConcreteAlignmentSampler(epsilon)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    else:
        raise ValueError(f"Unknown alignment sampling mode: {mode}")


class OutputSampler:

    @property
    @abstractmethod
    def mode(self):
        raise NotImplementedError("OutputSampler#mode")

    @abstractmethod
    def sample(self, sample_shape, bijector, stddev, dtype):
        raise NotImplementedError("OutputSampler#sample")


class DeterministicOutputSampler(OutputSampler):
    mode = "deterministic"

    def sample(self, sample_shape, bijector, stddev, dtype):
        mode = tf.zeros(shape=sample_shape, dtype=dtype)
        output, log_prob = bijector.sample_and_log_probability(mode)
        return output, log_prob


class StochasticOutputSampler(OutputSampler):
    mode = "stochastic"

    def sample(self, sample_shape, bijector, stddev, dtype):
        import tensorflow_probability as tfp
        sample = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(sample_shape, dtype=dtype),
                                                          scale_identity_multiplier=stddev).sample()
        output, log_prob = bijector.sample_and_log_probability(sample)
        return output, log_prob


def output_sampler_factory(mode):
    if mode == DeterministicOutputSampler.mode:
        return DeterministicOutputSampler()
    elif mode == StochasticOutputSampler.mode:
        return StochasticOutputSampler()
    else:
        raise ValueError(f"Unknown output sampling mode: {mode}")


class Search:

    @property
    @abstractmethod
    def beam_width(self):
        raise NotImplementedError("Search#beam_width")

    @property
    @abstractmethod
    def state_size(self):
        raise NotImplementedError("Search#state_size")

    @abstractmethod
    def zero_state(self, batch_size, dtype):
        raise NotImplementedError("Search#zero_state")

    def search(self, state, log_emit_and_shift_probs, input_lengths):
        raise NotImplementedError("Search#search")


class Sampler:

    @property
    @abstractmethod
    def state_size(self):
        raise NotImplementedError("Sampler#state_size")

    @abstractmethod
    def zero_state(self, batch_size, dtype):
        raise NotImplementedError("Sampler#zero_state")

    @abstractmethod
    def sample(self, search_state, sample_shape, emit_prob_logits, bijector, prev_attend_idx, input_lengths,
               inverse_temperature, stddev,
               dtype):
        raise NotImplementedError("Sampler#sample")


class JointSamplerState(namedtuple("JointSamplerState", ["time",
                                                         "search_state",
                                                         "log_output_prob_history"])):
    def finalize(self, beam_width):
        search_state = self.search_state.finalize(beam_width)
        # (W, U, 1)
        log_output_prob = tf.transpose(self.log_output_prob_history.stack(), perm=[1, 0, 2])
        if beam_width > 1:
            best_beam_index = tf.stack([search_state.beam_branch_history,
                                        tf.range(tf.shape(search_state.beam_branch_history)[0])],
                                       axis=1)
            # (U, 1)
            log_output_prob = tf.gather_nd(log_output_prob, best_beam_index)[:search_state.times + 1, :]
            log_output_prob = tf.expand_dims(log_output_prob, axis=0)  # preserve batch dim
        return TransitionBasedSamplerState(time=self.time,
                                           search_state=search_state,
                                           log_output_prob_history=log_output_prob)


class JointSampler(Sampler):

    def __init__(self, alignment_sampler: AlignmentSampler, output_sampler: OutputSampler, search_method: Search):
        self._alignment_sampler = alignment_sampler
        self._output_sampler = output_sampler
        self._search_method = search_method

    @property
    def state_size(self):
        return JointSamplerState(time=(),
                                 search_state=self._search_method.state_size,
                                 log_output_prob_history=[None, 1])

    def zero_state(self, batch_size, dtype):
        return JointSamplerState(time=0,
                                 search_state=self._search_method.zero_state(batch_size, dtype),
                                 log_output_prob_history=tf.TensorArray(tf.float32,
                                                                        size=0,
                                                                        dynamic_size=True,
                                                                        element_shape=[
                                                                            None, 1],
                                                                        name="log_output_prob_history"))

    def sample(self, state: JointSamplerState, sample_shape, emit_prob_logits, bijector, prev_attend_idx, input_lengths,
               inverse_temperature, stddev,
               dtype):
        batch_size = tf.shape(emit_prob_logits)[0]
        bijector = bijector.select_candidates(prev_attend_idx, input_lengths)
        outputs, log_output_prob = self._output_sampler.sample(sample_shape, bijector, stddev, dtype)

        log_emit_probs, log_shift_probs, prev_attend_idx = self._alignment_sampler.sample(emit_prob_logits,
                                                                                          inverse_temperature,
                                                                                          prev_attend_idx)

        log_emit_and_shift_probs = tf.stack([log_emit_probs, log_shift_probs], axis=-1)
        current_log_emit_and_shift_probs = tf.gather_nd(log_emit_and_shift_probs,
                                                        tf.stack([tf.range(batch_size), prev_attend_idx], axis=1))
        log_joint_prob = current_log_emit_and_shift_probs + log_output_prob

        emit_or_shift, log_emit_and_shift_probs, search_state = self._search_method.search(state.search_state,
                                                                                           log_joint_prob,
                                                                                           input_lengths)

        output = tf.gather_nd(outputs, tf.stack([tf.range(batch_size), emit_or_shift], axis=1))
        log_output_prob = tf.gather_nd(log_output_prob, tf.stack([tf.range(batch_size), emit_or_shift], axis=1))

        next_state = JointSamplerState(time=state.time + 1,
                                       search_state=search_state,
                                       log_output_prob_history=state.log_output_prob_history.write(
                                           state.time,
                                           tf.expand_dims(tf.cast(log_output_prob, dtype=tf.float32), axis=-1)))
        return output, next_state


class TransitionBasedSamplerState(namedtuple("TransitionBasedSamplerState", ["time",
                                                                             "search_state",
                                                                             "log_output_prob_history"])):
    def finalize(self, beam_width):
        log_output_prob = tf.transpose(self.log_output_prob_history.stack(), perm=[1, 0, 2])
        search_state = self.search_state.finalize(beam_width)
        if beam_width > 1:
            best_beam_index = tf.stack([search_state.beam_branch_history,
                                        tf.range(tf.shape(search_state.beam_branch_history)[0])],
                                       axis=1)
            # (U, 1)
            log_output_prob = tf.gather_nd(log_output_prob, best_beam_index)[:search_state.times + 1, :]
            log_output_prob = tf.expand_dims(log_output_prob, axis=0)  # preserve batch dim
        return TransitionBasedSamplerState(time=self.time,
                                           search_state=search_state,
                                           log_output_prob_history=log_output_prob)


class TransitionBasedSampler(Sampler):

    def __init__(self, alignment_sampler: AlignmentSampler, output_sampler: OutputSampler, search_method: Search):
        self._alignment_sampler = alignment_sampler
        self._output_sampler = output_sampler
        self._search_method = search_method

    @property
    def state_size(self):
        return TransitionBasedSamplerState(time=(),
                                           search_state=self._search_method.state_size,
                                           log_output_prob_history=[None, 1])

    def zero_state(self, batch_size, dtype):
        return TransitionBasedSamplerState(time=0,
                                           search_state=self._search_method.zero_state(batch_size, dtype),
                                           log_output_prob_history=tf.TensorArray(tf.float32,
                                                                                  size=0,
                                                                                  dynamic_size=True,
                                                                                  element_shape=[
                                                                                      None,
                                                                                      1],
                                                                                  name="log_output_prob_history"))

    def sample(self, state: TransitionBasedSamplerState, sample_shape, emit_prob_logits, bijector, prev_attend_idx,
               input_lengths, inverse_temperature, stddev,
               dtype):
        batch_size = tf.shape(emit_prob_logits)[0]

        log_emit_probs, log_shift_probs, prev_attend_idx = self._alignment_sampler.sample(emit_prob_logits,
                                                                                          inverse_temperature,
                                                                                          prev_attend_idx)

        log_emit_and_shift_probs = tf.stack([log_emit_probs, log_shift_probs], axis=-1)
        log_emit_and_shift_probs = tf.gather_nd(log_emit_and_shift_probs,
                                                tf.stack([tf.range(batch_size), prev_attend_idx], axis=1))

        emit_or_shift, log_emit_and_shift_probs, search_state = self._search_method.search(state.search_state,
                                                                                           log_emit_and_shift_probs,
                                                                                           input_lengths)
        if self._search_method.beam_width == 1:
            gather_idx = tf.stack([tf.zeros([batch_size], dtype=search_state.current_attend_idx.dtype),
                                   search_state.current_attend_idx],
                                  axis=1)
        else:
            gather_idx = tf.stack([search_state.beam_branch, search_state.current_attend_idx], axis=1)

        output, log_output_prob = self._output_sampler.sample(sample_shape,
                                                              bijector.select(gather_idx),
                                                              stddev, dtype)

        log_output_prob = tf.expand_dims(log_output_prob,
                                         axis=1) if log_output_prob.shape.ndims == 1 else log_output_prob

        next_state = TransitionBasedSamplerState(time=state.time + 1,
                                                 search_state=search_state,
                                                 log_output_prob_history=state.log_output_prob_history.write(state.time,
                                                                                                             tf.cast(
                                                                                                                 log_output_prob,
                                                                                                                 dtype=tf.float32)))

        return output, next_state


class SamplerOutput(namedtuple("AlignmentSampleOutput", ["output",
                                                         "next_attend_idx",
                                                         "log_emit_and_shift_probs",
                                                         "log_output_prob"])):
    pass


class BeamSearchState(namedtuple("BeamSearchState", ["time",
                                                     "times",
                                                     "current_attend_idx",
                                                     "log_prob_history",
                                                     "is_finished",
                                                     "attend_idx_history",
                                                     "beam_branch",
                                                     "beam_branch_history",
                                                     "log_emit_and_shift_probs_history"])):

    def select_cell_state(self, cell_state):
        beam_idx = tf.expand_dims(self.beam_branch, axis=-1)
        next_cell_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(c=tf.gather_nd(cs.c, beam_idx),
                                                               h=tf.gather_nd(cs.h, beam_idx)) for cs in cell_state])
        return next_cell_state

    def finalize(self, beam_width):
        assert beam_width > 1
        from ssnt_tts_tensorflow import extract_best_beam_branch
        # (U, W)
        beam_branch = self.beam_branch_history.stack()
        # (U, W)
        attend_idx_history = self.attend_idx_history.stack()
        # (W, U, 2)
        log_emit_and_shift_probs_history = tf.transpose(self.log_emit_and_shift_probs_history.stack(), perm=[1, 0, 2])

        unfinished_penalty = -1e4
        score = self.log_prob_history + tf.cast(tf.logical_not(self.is_finished), dtype=tf.float32) * unfinished_penalty

        best_final_branch = tf.argmax(score, axis=0, output_type=tf.int32)
        best_log_prob_history = self.log_prob_history[best_final_branch]
        best_is_finished = self.is_finished[best_final_branch]
        best_attend_idx = self.current_attend_idx[best_final_branch]
        best_times = self.times[best_final_branch]

        best_beam_branch_history, best_attend_idx_history = extract_best_beam_branch(best_final_branch, beam_branch,
                                                                                     attend_idx_history,
                                                                                     beam_width)

        best_beam_index = tf.stack([best_beam_branch_history, tf.range(tf.shape(best_beam_branch_history)[0])], axis=1)
        best_log_emit_and_shift_probs_history = tf.gather_nd(log_emit_and_shift_probs_history,
                                                             best_beam_index)

        return BeamSearchState(
            time=self.time,
            times=best_times,
            current_attend_idx=best_attend_idx,
            log_prob_history=best_log_prob_history,
            is_finished=best_is_finished,
            attend_idx_history=tf.expand_dims(best_attend_idx_history[:best_times + 1], axis=0),  # preserve batch dim
            beam_branch=beam_branch,
            beam_branch_history=best_beam_branch_history,
            log_emit_and_shift_probs_history=tf.expand_dims(best_log_emit_and_shift_probs_history[:best_times + 1, :],
                                                            axis=0))  # preserve batch dim

    def finalize_output(self, output):
        # output's shape is (W, U, D)
        best_beam_index = tf.stack([self.beam_branch_history, tf.range(tf.shape(self.beam_branch_history)[0])], axis=1)
        return tf.gather_nd(output, best_beam_index)[:self.times + 1, :]


class BeamSearch(Search):

    def __init__(self, beam_width, score_bias):
        self._beam_width = beam_width
        self._score_bias = score_bias

    @property
    def beam_width(self):
        return self._beam_width

    @property
    def state_size(self):
        return BeamSearchState(
            time=(),
            times=[self._beam_width],
            current_attend_idx=[self._beam_width],
            log_prob_history=[self._beam_width],
            is_finished=[self._beam_width],
            attend_idx_history=[self._beam_width],
            beam_branch=[self._beam_width],
            beam_branch_history=[self._beam_width],
            log_emit_and_shift_probs_history=[self._beam_width, 2])

    def zero_state(self, batch_size, dtype):
        return BeamSearchState(
            time=0,
            times=tf.zeros([self._beam_width], dtype=tf.int32),
            current_attend_idx=tf.zeros([self._beam_width], dtype=tf.int32),
            log_prob_history=tf.zeros([self._beam_width], dtype=tf.float32),
            is_finished=tf.zeros([self._beam_width], dtype=tf.bool),
            attend_idx_history=tf.TensorArray(tf.int32,
                                              size=0,
                                              dynamic_size=True,
                                              element_shape=[self._beam_width],
                                              name="attend_idx_history"),
            beam_branch=tf.zeros([self._beam_width], dtype=tf.int32),
            beam_branch_history=tf.TensorArray(tf.int32,
                                               size=0,
                                               dynamic_size=True,
                                               element_shape=[self._beam_width],
                                               name="beam_branch_history"),
            log_emit_and_shift_probs_history=tf.TensorArray(tf.float32,
                                                            size=0,
                                                            dynamic_size=True,
                                                            element_shape=[self._beam_width, 2],
                                                            name="log_emit_and_shift_probs_history"))

    def search(self, state: BeamSearchState,
               log_emit_and_shift_probs, input_lengths):
        from ssnt_tts_tensorflow import beam_search_decode

        prev_attend_idx = state.current_attend_idx
        score_bias = tf.expand_dims(tf.cast(tf.logical_not(state.is_finished), dtype=tf.float32),
                                    axis=-1) * self._score_bias
        scores = log_emit_and_shift_probs if self._score_bias == 0.0 else log_emit_and_shift_probs + score_bias

        emit_or_shift, log_prob_history, next_t, next_u, is_finished, beam_branch = beam_search_decode(
            scores,
            state.log_prob_history,
            state.is_finished,
            prev_attend_idx,
            state.times,
            tf.cast(input_lengths, dtype=tf.int32)[0],
            self._beam_width)

        next_beam_state = BeamSearchState(time=state.time + 1,
                                          times=next_u,
                                          current_attend_idx=next_t,
                                          log_prob_history=log_prob_history,
                                          is_finished=is_finished,
                                          attend_idx_history=state.attend_idx_history.write(state.time, next_t),
                                          beam_branch=beam_branch,
                                          beam_branch_history=state.beam_branch_history.write(state.time, beam_branch),
                                          log_emit_and_shift_probs_history=state.log_emit_and_shift_probs_history.write(
                                              state.time, tf.cast(log_emit_and_shift_probs, dtype=tf.float32)))

        return emit_or_shift, log_emit_and_shift_probs, next_beam_state


class SamplerMode:
    JOINT = "joint"
    TRANSITION = "transition"


def sampler_factory(mode, alignment_sampler, output_sampler, beam_width, score_bias):
    search_method = BeamSearch(beam_width, score_bias) if beam_width > 1 else GreedySearch()
    if mode == SamplerMode.TRANSITION:
        return TransitionBasedSampler(alignment_sampler, output_sampler, search_method)
    elif mode == SamplerMode.JOINT:
        return JointSampler(alignment_sampler, output_sampler, search_method)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")


class GreedySearchState(
    namedtuple("GreedySearchState", ["time",
                                     "current_attend_idx",
                                     "attend_idx_history",
                                     "log_emit_and_shift_probs_history"])):

    def select_cell_state(self, cell_state):
        return cell_state

    def finalize(self, beam_width):
        return GreedySearchState(
            time=self.time,
            current_attend_idx=self.current_attend_idx,
            attend_idx_history=tf.transpose(self.attend_idx_history.stack(), [1, 0]),
            log_emit_and_shift_probs_history=tf.transpose(self.log_emit_and_shift_probs_history.stack(),
                                                          perm=[1, 0, 2]))

    def finalize_output(self, output):
        return output


class GreedySearch(Search):
    def __init__(self):
        pass

    @property
    def beam_width(self):
        return 1

    @property
    def state_size(self):
        batch_size = None
        return GreedySearchState(
            time=(),
            current_attend_idx=[batch_size],
            attend_idx_history=[batch_size],
            log_emit_and_shift_probs_history=[batch_size, 2])

    def zero_state(self, batch_size, dtype):
        return GreedySearchState(
            time=0,
            current_attend_idx=tf.zeros([batch_size], dtype=tf.int32),
            attend_idx_history=tf.TensorArray(tf.int32,
                                              size=0,
                                              dynamic_size=True,
                                              element_shape=[None],
                                              name="attend_idx_history"),
            log_emit_and_shift_probs_history=tf.TensorArray(tf.float32,
                                                            size=0,
                                                            dynamic_size=True,
                                                            element_shape=[None, 2],
                                                            name="log_emit_and_shift_probs_history"))

    def search(self, state: GreedySearchState,
               log_emit_and_shift_probs, input_lengths):
        prev_attend_idx = state.current_attend_idx
        batch_size = tf.shape(state.current_attend_idx)[0]

        emit_or_shift = tf.argmax(
            log_emit_and_shift_probs,
            axis=-1,
            output_type=tf.int32)
        # Bound index to defined region
        next_attend_idx = tf.minimum(prev_attend_idx + emit_or_shift,
                                     tf.cast(input_lengths - 1, dtype=emit_or_shift.dtype))

        state = GreedySearchState(
            time=state.time + 1,
            current_attend_idx=next_attend_idx,
            attend_idx_history=state.attend_idx_history.write(state.time, next_attend_idx),
            log_emit_and_shift_probs_history=state.log_emit_and_shift_probs_history.write(
                state.time, tf.cast(log_emit_and_shift_probs, dtype=tf.float32)))

        return emit_or_shift, log_emit_and_shift_probs, state


class SSNTDecoderCellState(
    namedtuple("SSNTDecoderCellState", ["cell_state",
                                        "time",
                                        "sampler_state"])):

    def step(self, cell_state, sampler_state):
        return SSNTDecoderCellState(cell_state=sampler_state.search_state.select_cell_state(cell_state),
                                    time=self.time + 1,
                                    sampler_state=sampler_state)

    def finalize(self, beam_width):
        return SSNTDecoderCellState(cell_state=self.cell_state,
                                    time=self.time,
                                    sampler_state=self.sampler_state.finalize(beam_width))


class SSNTDecoderCellOutput(
    namedtuple("SSNTDecoderCellOutput", ["cell_output",
                                         "bijector",
                                         "mode_cell_output",
                                         "log_emit_probs",
                                         "log_shift_probs"])):

    def finalize_output(self, search_state):
        return search_state.finalize_output(self.mode_cell_output)


class SSNTDecoderCellCombinationMode:
    CONCAT = "concat"
    ADD = "add"


class SSNTDecoderCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, out_units, unconditional_layer_num_units,
                 conditional_layer_num_units,
                 memory, memory_lengths,
                 sampler: Sampler,
                 is_training,
                 is_validation,
                 bijector_config: BijectorConfig,
                 combination_mode=SSNTDecoderCellCombinationMode.CONCAT,
                 output_stddev=1.0, sigmoid_noise=0.0,
                 sigmoid_temperature=1.0, predict_sigmoid_temperature=False,
                 zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
                 beam_width=1,
                 lstm_impl=LSTMImpl.LSTMCell, epsilon=None,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(SSNTDecoderCell, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._memory_size = memory.get_shape()[-1].value
        self.batch_size = tf.shape(memory)[0]

        self._out_units = out_units
        self._num_units = unconditional_layer_num_units
        self._memory = memory
        self._memory_lengths = memory_lengths
        self._sampler = sampler
        self._combination_mode = combination_mode
        self._output_stddev = output_stddev
        self._sigmoid_noise = sigmoid_noise
        self._predict_sigmoid_temperature = predict_sigmoid_temperature
        self._inverse_sigmoid_temperature = 1.0 / sigmoid_temperature
        self._is_training = is_training
        self._epsilon = epsilon or backend.epsilon()
        self._is_validation = is_validation
        self._beam_width = beam_width

        self.alignment_layer = tf.layers.Dense(1, dtype=dtype)
        self.sigmoid_temperature_layer = tf.layers.Dense(1,
                                                         activation=tf.exp,
                                                         kernel_initializer=tf.initializers.zeros(),
                                                         dtype=dtype) if predict_sigmoid_temperature else None

        self._cell = tf.nn.rnn_cell.MultiRNNCell([
            ZoneoutLSTMCell(num_units, is_training, zoneout_factor_cell, zoneout_factor_output,
                            lstm_impl=lstm_impl, dtype=dtype)
            for num_units in unconditional_layer_num_units], state_is_tuple=True)

        self.conditional_projection_layers = [
            tf.layers.Dense(num_units, activation=tf.nn.tanh, name=f"conditional_projection-{i}",
                            dtype=dtype) for i, num_units in
            enumerate(conditional_layer_num_units)]

        self.bijection_layer = bijection_layer_factory(bijector_config.bijection_type,
                                                       out_units,
                                                       bijector_config.num_layers,
                                                       maf_bijector_num_hidden_layers=bijector_config.maf_bijector_num_hidden_layers,
                                                       maf_bijector_num_hidden_units=bijector_config.maf_bijector_num_hidden_units,
                                                       maf_bijector_num_blocks=bijector_config.maf_bijector_num_blocks,
                                                       maf_bijector_activation_function=bijector_config.maf_bijector_activation_function,
                                                       dtype=dtype)

    @property
    def state_size(self):
        return SSNTDecoderCellState(
            cell_state=self._cell.state_size,
            time=(),
            sampler_state=self._sampler.state_size)

    @property
    def output_size(self):
        return SSNTDecoderCellOutput(cell_output=self._cell.output_size,
                                     bijector=(),  # bijector is empty at prediction time
                                     mode_cell_output=self._out_units,
                                     log_emit_probs=tf.shape(self._memory)[:2],
                                     log_shift_probs=tf.shape(self._memory)[:2])

    def zero_state(self, batch_size, dtype):
        cell_state = self._cell.zero_state(batch_size, dtype)
        return SSNTDecoderCellState(
            cell_state=cell_state,
            time=tf.convert_to_tensor(0),
            sampler_state=self._sampler.zero_state(batch_size, dtype))

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

    def call(self, inputs, state: SSNTDecoderCellState):

        # (B, D_d)
        cell_output, next_cell_state = self._cell(inputs, state.cell_state)

        # (B, T, D_e + D_d) or (B, T, D_e == D_d)
        concatenated_cell_output = self.combine_input_and_feedback(cell_output)

        # (B, T, D_c)
        conditional_output = reduce(lambda acc, cp: cp(acc), self.conditional_projection_layers,
                                    concatenated_cell_output)

        alignment_inputs = conditional_output

        # (B, T)
        emit_prob_logits = tf.squeeze(self.alignment_layer(alignment_inputs), axis=-1)
        emit_prob_logits_for_sampling = emit_prob_logits
        if self._sigmoid_noise > 0.0 and self._is_training:
            emit_prob_logits = emit_prob_logits + random_logistic(tf.shape(emit_prob_logits),
                                                                  dtype=inputs.dtype,
                                                                  epsilon=self._epsilon)

        inverse_sigmoid_temperature = tf.squeeze(
            self.sigmoid_temperature_layer(-alignment_inputs),  # ToDo: Fix with exp(-dense(alignment_inputs))
            axis=-1) * self._inverse_sigmoid_temperature if self._predict_sigmoid_temperature else self._inverse_sigmoid_temperature

        emit_prob_logits = emit_prob_logits * inverse_sigmoid_temperature
        log_emit_probs = tf.math.log_sigmoid(emit_prob_logits)
        # log(1 - sigmoid(x)) = -x + log(sigmoid(x))
        log_shift_prob = - emit_prob_logits + log_emit_probs

        bijector = Bijector(self.bijection_layer, conditional_output,
                            output_stddev=self._output_stddev)

        # inference
        # (B, D)
        output, inference_state = self._sampler.sample(state.sampler_state, [self.batch_size, self._out_units],
                                                       emit_prob_logits_for_sampling, bijector,
                                                       state.sampler_state.search_state.current_attend_idx,
                                                       self._memory_lengths,
                                                       inverse_temperature=inverse_sigmoid_temperature,
                                                       stddev=self._output_stddev, dtype=inputs.dtype)

        next_state = state.step(next_cell_state, inference_state)

        # bijector is necessary only at training and validation to compute likelihood at the next layer
        return SSNTDecoderCellOutput(cell_output=conditional_output,
                                     bijector=bijector if self._is_training or self._is_validation else (),
                                     mode_cell_output=output,
                                     log_emit_probs=log_emit_probs,
                                     log_shift_probs=log_shift_prob), next_state

    def combine_input_and_feedback(self, cell_output):
        if self._beam_width > 1:
            # (B, T, D_d)
            cell_output = tf.tile(tf.expand_dims(cell_output, axis=1), multiples=[1, tf.shape(self._memory)[1], 1])
            memory = tf.tile(self._memory, multiples=[self._beam_width, 1, 1])
            # (B, T, D_e + D_d)
            concatenated_cell_output = tf.concat([memory, cell_output], axis=-1)
            return concatenated_cell_output
        if self._combination_mode == SSNTDecoderCellCombinationMode.CONCAT:
            # (B, T, D_d)
            cell_output = tf.tile(tf.expand_dims(cell_output, axis=1), multiples=[1, tf.shape(self._memory)[1], 1])
            # (B, T, D_e + D_d)
            concatenated_cell_output = tf.concat([self._memory, cell_output], axis=-1)
            return concatenated_cell_output
        elif self._combination_mode == SSNTDecoderCellCombinationMode.ADD:
            # (B, 1, D_d)
            cell_output = tf.expand_dims(cell_output, axis=1)
            # (B, T, D_d == D_e)
            added_cell_output = self._memory + cell_output
            return added_cell_output
        else:
            raise ValueError(f"Unknown combination mode: {self._combination_mode}")


class OutputProbabilityCellState(
    namedtuple("OutputProbabilityCellState",
               ["time", "log_forward_probs", "total_log_likelihood", "log_forward_prob_history"])):
    pass


class OutputProbabilityCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, target, target_lengths, memory_lengths, max_memory_length,
                 output_stddev,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(OutputProbabilityCell, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)

        self._target = target
        self._target_lengths = target_lengths
        self.batch_size = tf.shape(target)[0]
        self._memory_lengths = memory_lengths
        self._max_memory_length = max_memory_length
        self._output_stddev = output_stddev
        target_dim = target.get_shape()[-1].value
        self._log_normalization_constant = -0.5 * target_dim * (
                math.log(2. * math.pi, math.e) + 2. * math.log(output_stddev, math.e))

    def call(self, inputs, state: OutputProbabilityCellState):
        time = state.time
        prev_log_forward_probs = state.log_forward_probs
        total_log_likelihood = state.total_log_likelihood
        cell_output, bijector, mode_cell_output, log_emit_probs, log_shift_probs = inputs

        target = self._target[:, time, :]

        # (B, T)
        log_likelihood_y = bijector.log_probability(target)

        # (B, T)
        current_log_forward_emit = tf.cast(log_emit_probs, tf.float32) + prev_log_forward_probs
        current_log_forward_shift = tf.pad((tf.cast(log_shift_probs, tf.float32) + prev_log_forward_probs)[:, :-1],
                                           paddings=[[0, 0], [1, 0]], constant_values=-1e5)
        current_log_forward_probs = tf.cast(log_likelihood_y, tf.float32) + tf.reduce_logsumexp(
            tf.stack([current_log_forward_emit, current_log_forward_shift], axis=-1), axis=-1)

        # (B, T)
        total_log_likelihood_if_last = tf.where(tf.equal(self._target_lengths - 1, tf.to_int64(time)),
                                                tf.gather_nd(current_log_forward_probs,
                                                             tf.stack([tf.range(tf.to_int64(self.batch_size)),
                                                                       self._memory_lengths - 1], axis=1)),
                                                tf.zeros([self.batch_size], dtype=current_log_forward_probs.dtype))

        log_forward_prob_history = state.log_forward_prob_history.write(time, current_log_forward_probs)

        # total log likelihood requires high precision
        total_log_likelihood_if_last = tf.cast(total_log_likelihood_if_last, tf.float32) \
            if total_log_likelihood_if_last.dtype is not tf.float32 else total_log_likelihood_if_last

        next_state = OutputProbabilityCellState(time=time + 1,
                                                log_forward_probs=current_log_forward_probs,
                                                total_log_likelihood=total_log_likelihood + total_log_likelihood_if_last,
                                                log_forward_prob_history=log_forward_prob_history)
        return mode_cell_output, next_state

    @property
    def state_size(self):
        return OutputProbabilityCellState(time=(),
                                          log_forward_probs=[None, None],
                                          total_log_likelihood=[None],
                                          log_forward_prob_history=self._max_memory_length)

    @property
    def output_size(self):
        return self._target.get_shape()[-1].value

    def zero_state(self, batch_size, dtype):
        log_prob = tf.concat([tf.zeros([batch_size, 1], dtype=tf.float32),
                              tf.fill([batch_size, self._max_memory_length - 1], tf.constant(-1e10, dtype=tf.float32))],
                             axis=1)
        return OutputProbabilityCellState(time=0,
                                          log_forward_probs=log_prob,
                                          # total log likelihood requires high precision
                                          total_log_likelihood=tf.zeros([batch_size], tf.float32),
                                          log_forward_prob_history=tf.TensorArray(tf.float32,
                                                                                  size=0,
                                                                                  dynamic_size=True,
                                                                                  element_shape=[None, None],
                                                                                  name="log_forward_prob_history"))


class ValidationHelper(Helper):

    def __init__(self, targets, batch_size, output_dim, r, n_feed_frame=1, teacher_forcing=False):
        assert n_feed_frame <= r
        self._batch_size = batch_size
        self._output_dim = output_dim
        self._end_token = tf.tile([0.0], [output_dim * r])
        self.n_feed_frame = n_feed_frame
        self.num_steps = tf.shape(targets)[1] // r
        self.teacher_forcing = teacher_forcing
        self._targets = tf.reshape(targets,
                                   shape=tf.stack([self.batch_size, self.num_steps, tf.to_int32(output_dim * r)]))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (
            tf.tile([False], [self._batch_size]),
            _go_frames(self._batch_size, self._output_dim * self.n_feed_frame, self._targets.dtype))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        finished = (time + 1 >= self.num_steps)
        next_inputs = self._targets[:, time,
                      -self._output_dim * self.n_feed_frame:] if self.teacher_forcing else outputs[:,
                                                                                           -self._output_dim * self.n_feed_frame:]
        next_inputs.set_shape([outputs.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return (finished, next_inputs, state)


class TrainingHelper(Helper):

    def __init__(self, targets, output_dim, r, n_feed_frame=1):
        assert n_feed_frame <= r
        t_shape = tf.shape(targets)
        self._batch_size = t_shape[0]
        self._output_dim = output_dim
        self.n_feed_frame = n_feed_frame

        self._targets = tf.reshape(targets,
                                   shape=tf.stack([self.batch_size, t_shape[1] // r, tf.to_int32(output_dim * r)]))
        self._targets.set_shape((targets.get_shape()[0].value, None, output_dim * r))

        # Use full length for every target because we don't want to mask the padding frames
        num_steps = tf.shape(self._targets)[1]
        self._lengths = tf.tile([num_steps], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (
            tf.tile([False], [self._batch_size]),
            _go_frames(self._batch_size, self._output_dim * self.n_feed_frame, self._targets.dtype))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        finished = (time + 1 >= self._lengths)
        next_inputs = self._targets[:, time, -self._output_dim * self.n_feed_frame:]
        next_inputs.set_shape([outputs.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return (finished, next_inputs, state)


class SSNTInferenceHelper(Helper):

    def __init__(self, batch_size, memory_sequence_length, output_dim, r, dtype, n_feed_frame=1, min_iters=10):
        assert n_feed_frame <= r
        self._batch_size = batch_size
        self._memory_sequence_length = memory_sequence_length
        self._output_dim = output_dim
        self._dtype = dtype
        self.n_feed_frame = n_feed_frame
        self.min_iters = min_iters

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (
            tf.tile([False], [self._batch_size]),
            _go_frames(self._batch_size, self._output_dim * self.n_feed_frame, self._dtype))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._batch_size])

    def next_inputs(self, time, outputs: SSNTDecoderCellOutput, state: Tuple[SSNTDecoderCellState], sample_ids,
                    name=None):
        state0 = state[0]
        finished = self.is_finished(state0.sampler_state.search_state.current_attend_idx, time)
        next_input = outputs.mode_cell_output[:, -self._output_dim * self.n_feed_frame:]
        next_input.set_shape([next_input.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return finished, next_input, state

    def is_finished(self, current_attend_idx, time):
        termination_criteria = tf.greater_equal(tf.to_int64(current_attend_idx), self._memory_sequence_length - 1)
        minimum_requirement = tf.greater(time, self.min_iters)
        termination = tf.logical_and(termination_criteria, minimum_requirement)
        return tf.reduce_all(termination, axis=0)


class SSNTBeamSearchHelper(Helper):

    def __init__(self, beam_width, memory_sequence_length, output_dim, r, dtype, n_feed_frame=1, min_iters=10):
        assert n_feed_frame <= r
        self._beam_width = beam_width
        self._memory_sequence_length = memory_sequence_length
        self._output_dim = output_dim
        self._dtype = dtype
        self.n_feed_frame = n_feed_frame
        self.min_iters = min_iters

    @property
    def batch_size(self):
        return self._beam_width

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return (
            tf.tile([False], [self._beam_width]),
            _go_frames(self._beam_width, self._output_dim * self.n_feed_frame, self._dtype))

    def sample(self, time, outputs, state, name=None):
        # return all-zero dummy tensor
        return tf.tile([0], [self._beam_width])

    def next_inputs(self, time, outputs: SSNTDecoderCellOutput, state: Tuple[SSNTDecoderCellState], sample_ids,
                    name=None):
        state0 = state[0]
        finished = self.is_finished(state0.sampler_state.search_state.current_attend_idx, time)
        next_input = outputs.mode_cell_output[:, -self._output_dim * self.n_feed_frame:]
        next_input.set_shape([next_input.get_shape()[0].value, self._output_dim * self.n_feed_frame])
        return finished, next_input, state

    def is_finished(self, current_attend_idx, time):
        termination_criteria = tf.greater_equal(tf.to_int64(current_attend_idx), self._memory_sequence_length - 1)
        minimum_requirement = tf.greater(time, self.min_iters)
        termination = tf.logical_and(termination_criteria, minimum_requirement)
        return tf.reduce_all(termination, axis=0)


def _go_frames(batch_size, output_dim, dtype):
    return tf.tile(tf.convert_to_tensor([[0.0]], dtype=dtype), [batch_size, output_dim])


class SSNTDecoder(tf.layers.Layer):

    def __init__(self,
                 num_mels,
                 unconditional_layer_num_units,
                 conditional_layer_num_units,
                 prenet_out_units=(256, 128),
                 alignment_sampling_mode=DeterministicAlignmentSampler.mode,
                 output_sampling_mode=DeterministicOutputSampler.mode,
                 sampling_mode=SamplerMode.TRANSITION,
                 bijector_config=BijectorConfig(bijection_type=BijectionType.IDENTITY,
                                                num_layers=2,
                                                maf_bijector_num_hidden_layers=2,
                                                maf_bijector_num_hidden_units=640,
                                                maf_bijector_num_blocks=160,
                                                maf_bijector_activation_function=tf.nn.relu),
                 combination_mode=SSNTDecoderCellCombinationMode.CONCAT,
                 output_stddev=1.0,
                 sigmoid_noise=0.0,
                 sigmoid_temperature=1.0,
                 predict_sigmoid_temperature=False,
                 outputs_per_step=2,
                 n_feed_frame=1,
                 max_iters=500,
                 drop_rate=0.5,
                 zoneout_factor_cell=0.0,
                 zoneout_factor_output=0.0,
                 beam_width=1,
                 beam_search_score_bias=0.0,
                 lstm_impl=LSTMImpl.LSTMCell,
                 trainable=True, name=None, dtype=None, **kwargs):
        super(SSNTDecoder, self).__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self._num_mels = num_mels
        self._out_units = outputs_per_step * num_mels
        self._unconditional_layer_num_units = unconditional_layer_num_units
        self._conditional_layer_num_units = conditional_layer_num_units
        self._prenet_out_units = prenet_out_units
        alignment_sampler = alignment_sampler_factory(alignment_sampling_mode,
                                                      distribution="binconcrete" if sigmoid_noise > 0.0 else "logistic",
                                                      epsilon=backend.epsilon())
        output_sampler = output_sampler_factory(output_sampling_mode)
        self._sampler = sampler_factory(sampling_mode, alignment_sampler, output_sampler,
                                        beam_width, beam_search_score_bias)
        self._bijector_config = bijector_config
        self._combination_mode = combination_mode
        self._output_stddev = output_stddev
        self._sigmoid_noise = sigmoid_noise
        self._sigmoid_temperature = sigmoid_temperature
        self._predict_sigmoid_temperature = predict_sigmoid_temperature
        self._outputs_per_step = outputs_per_step
        self._n_feed_frame = n_feed_frame
        self._max_iters = max_iters
        self._beam_width = beam_width
        self._drop_rate = drop_rate
        self._zoneout_factor_cell = zoneout_factor_cell
        self._zoneout_factor_output = zoneout_factor_output
        self._lstm_impl = lstm_impl

    def call(self, memory, target=None, is_training=None, is_validation=None,
             teacher_forcing=False, memory_sequence_length=None, target_sequence_length=None,
             apply_dropout_on_inference=None):
        batch_size = tf.shape(memory)[0]

        prenets = tuple([PreNet(out_unit, is_training, self._drop_rate, apply_dropout_on_inference, dtype=self.dtype)
                         for out_unit in self._prenet_out_units])

        decoder_layer = SSNTDecoderCell(self._out_units,
                                        self._unconditional_layer_num_units,
                                        self._conditional_layer_num_units,
                                        memory, memory_sequence_length,
                                        sampler=self._sampler,
                                        is_training=is_training,
                                        is_validation=is_validation,
                                        bijector_config=self._bijector_config,
                                        combination_mode=self._combination_mode,
                                        output_stddev=self._output_stddev,
                                        sigmoid_noise=self._sigmoid_noise,
                                        sigmoid_temperature=self._sigmoid_temperature,
                                        predict_sigmoid_temperature=self._predict_sigmoid_temperature,
                                        zoneout_factor_cell=self._zoneout_factor_cell,
                                        zoneout_factor_output=self._zoneout_factor_output,
                                        beam_width=self._beam_width,
                                        lstm_impl=self._lstm_impl,
                                        dtype=self.dtype)

        reduced_target = tf.reshape(target,
                                    shape=tf.stack(
                                        [batch_size, tf.shape(target)[1] // self._outputs_per_step, self._out_units]))
        reduced_target_lengths = target_sequence_length // self._outputs_per_step if is_training or is_validation else None
        output_cell = OutputProbabilityCell(reduced_target, reduced_target_lengths, memory_sequence_length,
                                            tf.shape(memory)[1],
                                            self._output_stddev,
                                            dtype=self.dtype) if is_training or is_validation else None

        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [
                DecoderPreNetWrapper(decoder_layer, prenets),
                output_cell
            ] if is_training or is_validation else [
                DecoderPreNetWrapper(decoder_layer, prenets)
            ])

        decoder_initial_state = decoder_cell.zero_state(self._beam_width, dtype=memory.dtype) if self._beam_width > 1 \
            else decoder_cell.zero_state(batch_size, dtype=memory.dtype)

        helper = TrainingHelper(target,
                                self._num_mels,
                                self._outputs_per_step,
                                n_feed_frame=self._n_feed_frame) if is_training \
            else ValidationHelper(target, batch_size,
                                  self._num_mels,
                                  self._outputs_per_step,
                                  n_feed_frame=self._n_feed_frame,
                                  teacher_forcing=teacher_forcing) if is_validation \
            else SSNTBeamSearchHelper(self._beam_width, memory_sequence_length,
                                      self._num_mels,
                                      self._outputs_per_step,
                                      dtype=memory.dtype,
                                      n_feed_frame=self._n_feed_frame) if self._beam_width > 1 \
            else SSNTInferenceHelper(batch_size, memory_sequence_length,
                                     self._num_mels,
                                     self._outputs_per_step,
                                     dtype=memory.dtype,
                                     n_feed_frame=self._n_feed_frame)

        (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
            BasicDecoder(decoder_cell, helper, decoder_initial_state),
            maximum_iterations=self._max_iters,
            swap_memory=True)

        final_decoder_state = tuple([final_decoder_state[0].finalize(self._beam_width),
                                     final_decoder_state[1]]) if is_training or is_validation else tuple(
            [s.finalize(self._beam_width) for s in final_decoder_state])
        decoder_output = decoder_outputs if is_training or is_validation else decoder_outputs.finalize_output(
            final_decoder_state[0].sampler_state.search_state)

        mel_output = tf.reshape(decoder_output, [batch_size, -1, self._num_mels])
        return mel_output, final_decoder_state
