# ==============================================================================
# Copyright (c) 2018-2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf
from tensorflow.python.keras import backend
from core.modules import Embedding
from core.learning_rate_decay import learning_rate_decay_factory
from core.encoder_factories import encoder_factory
from core.ssnt import SSNTDecoder
from core.metrics import SSNTMetricsSaver
from core.flow import BijectorConfig
from utils.alignment_error_detector import detect_alignment_error


class SSNTModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL
            is_prediction = mode == tf.estimator.ModeKeys.PREDICT

            embedding = Embedding(params.num_symbols, embedding_dim=params.embedding_dim,
                                  output_dtype=tf.float16 if params.half_precision else tf.float32,
                                  dtype=backend.floatx())

            if params.use_accent_type:
                accent_embedding = Embedding(params.num_accent_type,
                                             embedding_dim=params.accent_type_embedding_dim,
                                             index_offset=params.accent_type_offset,
                                             output_dtype=tf.float16 if params.half_precision else tf.float32,
                                             dtype=backend.floatx())

            encoder = encoder_factory(params, is_training)

            decoder = SSNTDecoder(prenet_out_units=params.decoder_prenet_out_units,
                                  drop_rate=params.decoder_prenet_drop_rate,
                                  unconditional_layer_num_units=params.ssnt_unconditional_layer_num_units,
                                  conditional_layer_num_units=params.ssnt_conditional_layer_num_units,
                                  num_mels=params.num_mels,
                                  alignment_sampling_mode=params.ssnt_alignment_sampling_mode,
                                  output_sampling_mode=params.ssnt_output_sampling_mode,
                                  sampling_mode=params.ssnt_sampling_mode,
                                  bijector_config=BijectorConfig(bijection_type=params.ssnt_bijection_type,
                                                                 num_layers=params.ssnt_num_bijection_layers,
                                                                 maf_bijector_num_hidden_layers=params.ssnt_maf_bijector_num_hidden_layers,
                                                                 maf_bijector_num_hidden_units=params.ssnt_maf_bijector_num_hidden_units,
                                                                 maf_bijector_num_blocks=params.ssnt_maf_bijector_num_blocks,
                                                                 maf_bijector_activation_function=params.ssnt_maf_bijector_activation_function),
                                  combination_mode=params.ssnt_input_and_feedback_combination_mode,
                                  output_stddev=params.ssnt_output_stddev,
                                  sigmoid_noise=params.ssnt_sigmoid_noise,
                                  sigmoid_temperature=params.ssnt_sigmoid_temperature,
                                  predict_sigmoid_temperature=params.ssnt_predict_sigmoid_temperature,
                                  outputs_per_step=params.outputs_per_step,
                                  max_iters=params.max_iters,
                                  n_feed_frame=params.n_feed_frame,
                                  zoneout_factor_cell=params.zoneout_factor_cell,
                                  zoneout_factor_output=params.zoneout_factor_output,
                                  beam_width=params.ssnt_beam_width,
                                  beam_search_score_bias=params.ssnt_beam_search_score_bias,
                                  lstm_impl=params.lstm_impl,
                                  dtype=backend.floatx())

            target = labels.mel if (is_training or is_validation) else features.mel

            embedding_output = embedding(features.source)
            encoder_output = encoder(
                (embedding_output, accent_embedding(features.accent_type)),
                input_lengths=features.source_length) if params.use_accent_type else encoder(
                embedding_output, input_lengths=features.source_length)

            mel_output, decoder_state = decoder(encoder_output,
                                                is_training=is_training,
                                                is_validation=is_validation or params.use_forced_alignment_mode,
                                                teacher_forcing=params.use_forced_alignment_mode,
                                                memory_sequence_length=features.source_length,
                                                target_sequence_length=labels.target_length if not is_prediction else None,
                                                target=target,
                                                apply_dropout_on_inference=params.apply_dropout_on_inference)

            forward_prob = tf.transpose(decoder_state[1].log_forward_prob_history.stack(),
                                        [1, 2, 0]) if is_training or is_validation else None
            alignment = tf.one_hot(decoder_state[0].sampler_state.search_state.attend_idx_history,
                                   tf.shape(encoder_output)[1], axis=1)

            total_log_likelihood = decoder_state[1].total_log_likelihood if is_training or is_validation else None
            log_emit_and_shift_probs = decoder_state[0].sampler_state.search_state.log_emit_and_shift_probs_history
            log_output_prob = decoder_state[0].sampler_state.log_output_prob_history

            if params.use_forced_alignment_mode:
                mel_output, stop_token, decoder_state = decoder(encoder_output,
                                                                is_training=is_training,
                                                                is_validation=True,
                                                                teacher_forcing=False,
                                                                memory_sequence_length=features.source_length,
                                                                target_sequence_length=labels.target_length if not is_prediction else None,
                                                                target=target,
                                                                apply_dropout_on_inference=params.apply_dropout_on_inference)

                forward_prob = tf.transpose(decoder_state[1].log_forward_prob_history.stack(),
                                            [1, 2, 0]) if is_training or is_validation else None
                alignment = tf.one_hot(decoder_state[0].sampler_state.search_state.attend_idx_history,
                                       tf.shape(encoder_output)[1])

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                loss = tf.losses.compute_weighted_loss(-total_log_likelihood)

            if is_training:
                decay_fn = learning_rate_decay_factory(params)
                lr = decay_fn(global_step)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=backend.epsilon())

                gradients, variables = zip(*optimizer.compute_gradients(loss))

                clipped_gradients, gradient_global_norm = tf.clip_by_global_norm(gradients,
                                                                                 clip_norm=params.gradient_clip_norm)

                self.add_training_stats(loss=loss,
                                        learning_rate=lr,
                                        gradient_global_norm=gradient_global_norm)
                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                    summary_writer = tf.summary.FileWriter(model_dir)
                    alignment_saver = SSNTMetricsSaver([alignment], [tf.exp(forward_prob)],
                                                       [log_emit_and_shift_probs],
                                                       [log_output_prob],
                                                       global_step,
                                                       mel_output, mel_output, labels.mel,
                                                       labels.target_length,
                                                       features.id,
                                                       features.text,
                                                       params.alignment_save_steps,
                                                       mode, summary_writer,
                                                       save_training_time_metrics=params.save_training_time_metrics,
                                                       keep_eval_results_max_epoch=params.keep_eval_results_max_epoch)
                    hooks = [alignment_saver]
                    if params.record_profile:
                        profileHook = tf.train.ProfilerHook(save_steps=params.profile_steps, output_dir=model_dir,
                                                            show_dataflow=True, show_memory=True)
                        hooks.append(profileHook)
                    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                      training_hooks=hooks)

            if is_validation:
                # validation with teacher forcing
                mel_output_with_teacher, decoder_state_with_teacher = decoder(encoder_output,
                                                                              is_training=is_training,
                                                                              is_validation=is_validation,
                                                                              memory_sequence_length=features.source_length,
                                                                              target_sequence_length=labels.target_length,
                                                                              target=target,
                                                                              teacher_forcing=True,
                                                                              apply_dropout_on_inference=params.apply_dropout_on_inference)

                forward_prob_with_teacher = tf.transpose(decoder_state_with_teacher[1].log_forward_prob_history.stack(),
                                                         [1, 2, 0])
                alignment_with_teacher = tf.one_hot(
                    decoder_state_with_teacher[0].sampler_state.search_state.attend_idx_history,
                    tf.shape(encoder_output)[1], axis=1)

                log_emit_and_shift_probs_with_teacher = decoder_state_with_teacher[0].sampler_state.search_state.log_emit_and_shift_probs_history
                log_output_prob_with_teacher = decoder_state_with_teacher[0].sampler_state.log_output_prob_history

                total_log_likelihood_with_teacher = decoder_state_with_teacher[1].total_log_likelihood

                loss_with_teacher = tf.losses.compute_weighted_loss(-total_log_likelihood_with_teacher)

                eval_metric_ops = self.get_validation_metrics(loss_with_teacher=loss_with_teacher,
                                                              alignment=alignment)

                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = SSNTMetricsSaver(
                    [alignment, alignment_with_teacher], [tf.exp(forward_prob), tf.exp(forward_prob_with_teacher)],
                    [log_emit_and_shift_probs, log_emit_and_shift_probs_with_teacher],
                    [log_output_prob, log_output_prob_with_teacher],
                    global_step,
                    mel_output,
                    mel_output_with_teacher,
                    labels.mel,
                    labels.target_length,
                    features.id,
                    features.text,
                    1,
                    mode, summary_writer,
                    save_training_time_metrics=params.save_training_time_metrics,
                    keep_eval_results_max_epoch=params.keep_eval_results_max_epoch)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  evaluation_hooks=[alignment_saver],
                                                  eval_metric_ops=eval_metric_ops)

            if is_prediction:
                predictions = {
                    "id": features.id,
                    "key": features.key,
                    "mel": mel_output,
                    "mel_postnet": None,
                    "ground_truth_mel": features.mel,
                    "alignment": alignment,
                    "alignment2": None,
                    "alignment3": None,
                    "alignment4": None,
                    "alignment5": None,
                    "source": features.source,
                    "text": features.text,
                    "accent_type": features.accent_type if params.use_accent_type else None,
                    "log_emit_and_shift_probs": log_emit_and_shift_probs,
                    "log_output_prob": log_output_prob,
                }
                predictions = dict(filter(lambda xy: xy[1] is not None, predictions.items()))
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        super(SSNTModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def add_training_stats(loss, learning_rate, gradient_global_norm):
        if loss is not None:
            tf.summary.scalar("loss_with_teacher", loss)
        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("gradient_global_norm", gradient_global_norm)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(loss_with_teacher, alignment):
        metrics = {}
        if loss_with_teacher is not None:
            metrics["loss_with_teacher"] = tf.metrics.mean(loss_with_teacher)
        metrics["alignment_error_rate"] = tf.metrics.mean(detect_alignment_error(alignment[0]))
        return metrics


def model_factory(hparams, model_dir, run_config, warm_start_from=None):
    if hparams.model == "SSNTModel":
        model = SSNTModel(hparams, model_dir, config=run_config, warm_start_from=warm_start_from)
    else:
        raise ValueError(f"Unknown Tacotron model: {hparams.tacotron_model}")
    return model