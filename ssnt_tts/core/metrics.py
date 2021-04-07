# ==============================================================================
# Copyright (c) 2018-2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
""" Hooks. """

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import os
import numpy as np
import re
from typing import List, Iterable


def bytes_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
    assert isinstance(value, Iterable)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tfrecord(example: tf.train.Example, filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())


def write_training_result(global_step: int, id: List[int], text: List[str], predicted_mel: List[np.ndarray],
                          ground_truth_mel: List[np.ndarray], mel_length: List[int], alignment: List[np.ndarray],
                          filename: str):
    batch_size = len(ground_truth_mel)
    raw_predicted_mel = [m.tostring() for m in predicted_mel]
    raw_ground_truth_mel = [m.tostring() for m in ground_truth_mel]
    mel_width = ground_truth_mel[0].shape[1]
    padded_mel_length = [m.shape[0] for m in ground_truth_mel]
    predicted_mel_length = [m.shape[0] for m in predicted_mel]
    raw_alignment = [a.tostring() for a in alignment]
    alignment_source_length = [a.shape[1] for a in alignment]
    alignment_target_length = [a.shape[2] for a in alignment]
    example = tf.train.Example(features=tf.train.Features(feature={
        'global_step': int64_feature([global_step]),
        'batch_size': int64_feature([batch_size]),
        'id': int64_feature(id),
        'text': bytes_feature(text),
        'predicted_mel': bytes_feature(raw_predicted_mel),
        'ground_truth_mel': bytes_feature(raw_ground_truth_mel),
        'mel_length': int64_feature(padded_mel_length),
        'mel_length_without_padding': int64_feature(mel_length),
        'predicted_mel_length': int64_feature(predicted_mel_length),
        'mel_width': int64_feature([mel_width]),
        'alignment': bytes_feature(raw_alignment),
        'alignment_source_length': int64_feature(alignment_source_length),
        'alignment_target_length': int64_feature(alignment_target_length),
    }))
    write_tfrecord(example, filename)


def write_prediction_result(id: int, key: str, text: str,
                            log_emit_and_shift_prob: np.ndarray,
                            log_output_probs: np.ndarray,
                            filename: str):
    raw_log_emit_prob = log_emit_and_shift_prob[:, 0].tostring()
    raw_log_shift_prob = log_emit_and_shift_prob[:, 1].tostring()
    raw_log_output_probs = log_output_probs.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([id]),
        'key': bytes_feature([key.encode('utf-8')]),
        'text': bytes_feature([text.encode('utf-8')]),
        'log_emit_prob': bytes_feature([raw_log_emit_prob]),
        'log_shift_prob': bytes_feature([raw_log_shift_prob]),
        'log_output_probs': bytes_feature([raw_log_output_probs]),
    }))
    write_tfrecord(example, filename)


def save_alignment_and_log_probs(alignments,
                                 log_emit_and_shift_probs,
                                 log_output_probs,
                                 forward_probs,
                                 text, _id, global_step, path):
    num_forward_probs = len(forward_probs)
    num_alignment = len(alignments)
    num_log_emit_and_shift_probs = len(log_emit_and_shift_probs)
    num_log_output_probs = len(log_output_probs)
    n_subplot = num_forward_probs + num_alignment + num_log_emit_and_shift_probs + num_log_output_probs
    assert (num_forward_probs == num_alignment
            and num_alignment == num_log_emit_and_shift_probs
            and num_log_emit_and_shift_probs == num_log_output_probs)
    fig = plt.figure(figsize=(12, 20))

    for i, (alignment,
            forward_prob,
            log_emit_and_shift_prob,
            log_output_prob) in enumerate(zip(alignments,
                                              forward_probs,
                                              log_emit_and_shift_probs,
                                              log_output_probs)):
        ax = fig.add_subplot(n_subplot, 1, i * 4 + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none',
            cmap='binary')
        fig.colorbar(im, cax=inset_axes(ax, loc='upper left', width="15%", height="5%"), orientation="horizontal")
        ax.set_xlabel('Decoder timestep')
        ax.set_ylabel('Encoder timestep')
        ax.set_title(f"alignment{'' if i % 2 == 0 else ' (teacher forcing)'}")

        ax = fig.add_subplot(n_subplot, 1, i * 4 + 2)
        x = np.arange(log_emit_and_shift_prob.shape[0])
        ax.plot(x, log_emit_and_shift_prob[:, 0], label="emit")
        ax.plot(x, log_emit_and_shift_prob[:, 1], label="shift")
        ax.set_xlim(0, len(log_emit_and_shift_prob[:, 0]))
        ax.legend(loc='lower right')
        ax.set_xlabel('Decoder timestep')
        ax.set_ylabel('log probability')
        ax.set_title(f"alignment probability{'' if i % 2 == 0 else ' (teacher forcing)'}")

        ax = fig.add_subplot(n_subplot, 1, i * 4 + 3)
        ax.plot(x, log_output_prob[:, 0], label="out0")
        # ax.plot(x, log_output_prob[:, 1], label="out1")
        ax.set_xlim(0, len(log_output_prob[:, 0]))
        ax.set_xlabel('Decoder timestep')
        ax.set_ylabel('log probability')
        ax.set_title(f"output probability{'' if i % 2 == 0 else ' (teacher forcing)'}")

        ax = fig.add_subplot(n_subplot, 1, i * 4 + 4)
        if forward_prob is not None:
            im = ax.imshow(
                forward_prob,
                aspect='auto',
                origin='lower',
                interpolation='none',
                cmap='jet',
                norm=colors.LogNorm(vmin=1e-10, vmax=1.0))
            fig.colorbar(im, cax=inset_axes(ax, loc='upper left', width="15%", height="5%"), orientation="horizontal")
            ax.set_xlabel('Decoder timestep')
            ax.set_ylabel('Encoder timestep')
            ax.set_title(f"log forward probability {'' if i % 2 == 0 else ' (teacher forcing)'}")
    fig.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(path, format='png')
    plt.close()


def plot_mel(mel, mel_predicted, mel_predicted_with_teacher, text, _id, global_step, filename):
    from matplotlib import pylab as plt
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(3, 1, 1)
    im = ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax.set_title("ground truth")
    ax = fig.add_subplot(3, 1, 2)
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax.set_title("predicted")
    ax = fig.add_subplot(3, 1, 3)
    im = ax.imshow(mel_predicted_with_teacher.T, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax.set_title("predicted (teacher forcing)")
    fig.suptitle(f"record ID: {_id}\nglobal step: {global_step}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


class SSNTMetricsSaver(tf.train.SessionRunHook):

    def __init__(self, alignment_tensors, forward_probs,
                 log_emit_and_shift_probs_tensor, log_output_prob_tensor,
                 global_step_tensor,
                 predicted_mel_tensor, predicted_mel_with_teacher_tensor,
                 ground_truth_mel_tensor,
                 mel_length_tensor, id_tensor,
                 text_tensor, save_steps,
                 mode, writer: tf.summary.FileWriter,
                 save_training_time_metrics=True,
                 keep_eval_results_max_epoch=10):
        self.alignment_tensors = alignment_tensors
        self.forward_probs = forward_probs
        self.log_emit_and_shift_probs_tensor = log_emit_and_shift_probs_tensor
        self.log_output_prob_tensor = log_output_prob_tensor
        self.global_step_tensor = global_step_tensor
        self.predicted_mel_tensor = predicted_mel_tensor
        self.predicted_mel_with_teacher_tensor = predicted_mel_with_teacher_tensor
        self.ground_truth_mel_tensor = ground_truth_mel_tensor
        self.mel_length_tensor = mel_length_tensor
        self.id_tensor = id_tensor
        self.text_tensor = text_tensor
        self.save_steps = save_steps
        self.mode = mode
        self.writer = writer
        self.save_training_time_metrics = save_training_time_metrics
        self.keep_eval_results_max_epoch = keep_eval_results_max_epoch
        self.checkpoint_pattern = re.compile('all_model_checkpoint_paths: "model.ckpt-(\d+)"')

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor
        })

    def after_run(self,
                  run_context,
                  run_values):
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, alignments, forward_probs, log_emit_and_shift_probs, log_output_prob, predicted_mels, predicted_mels_with_teacher, ground_truth_mels, mel_length, ids, texts = run_context.session.run(
                (self.global_step_tensor, self.alignment_tensors, self.forward_probs,
                 self.log_emit_and_shift_probs_tensor, self.log_output_prob_tensor,
                 self.predicted_mel_tensor, self.predicted_mel_with_teacher_tensor,
                 self.ground_truth_mel_tensor, self.mel_length_tensor, self.id_tensor, self.text_tensor))
            alignments = [a.astype(np.float32) for a in alignments]
            forward_probs = [a.astype(np.float32) for a in forward_probs]
            log_emit_and_shift_probs = [p.astype(np.float32) for p in log_emit_and_shift_probs]
            log_output_prob = [p.astype(np.float32) for p in log_output_prob]
            predicted_mels = [m.astype(np.float32) for m in list(predicted_mels)]
            predicted_mels_with_teacher = [m.astype(np.float32) for m in list(predicted_mels_with_teacher)]
            ground_truth_mels = [m.astype(np.float32) for m in list(ground_truth_mels)]
            if self.mode == tf.estimator.ModeKeys.EVAL or self.save_training_time_metrics:
                id_strings = ",".join([str(i) for i in ids][:10])
                result_filename = "{}_result_step{:09d}_{}.tfrecord".format(self.mode, global_step_value, id_strings)
                tf.logging.info("Saving a %s result for %d at %s", self.mode, global_step_value, result_filename)
                write_training_result(global_step_value, list(ids), list(texts), predicted_mels,
                                      ground_truth_mels, list(mel_length),
                                      alignments + forward_probs,
                                      filename=os.path.join(self.writer.get_logdir(), result_filename))
            if self.mode == tf.estimator.ModeKeys.EVAL:
                alignments = [[a[i] for a in alignments] for i in range(alignments[0].shape[0])]
                forward_probs = [[a[i] for a in forward_probs] for i in range(forward_probs[0].shape[0])]
                log_emit_and_shift_probs = [[a[i] for a in log_emit_and_shift_probs] for i in
                                            range(log_emit_and_shift_probs[0].shape[0])]
                log_output_prob = [[a[i] for a in log_output_prob] for i in range(log_output_prob[0].shape[0])]
                for _id, text, align, forward_probs, log_emit_and_shift_probs, log_output_prob, pred_mel, pred_mel_wt, gt_mel in zip(
                        ids, texts, alignments,
                        forward_probs,
                        log_emit_and_shift_probs,
                        log_output_prob,
                        predicted_mels,
                        predicted_mels_with_teacher,
                        ground_truth_mels):
                    output_filename = "{}_result_step{:09d}_{:d}.png".format(self.mode,
                                                                             global_step_value, _id)
                    plot_mel(gt_mel, pred_mel, pred_mel_wt,
                             text.decode('utf-8'), _id, global_step_value,
                             os.path.join(self.writer.get_logdir(), "mel_" + output_filename))

    def end(self, session):
        if self.mode == tf.estimator.ModeKeys.EVAL:
            current_global_step = session.run(self.global_step_tensor)
            with open(os.path.join(self.writer.get_logdir(), "checkpoint")) as f:
                checkpoints = [ckpt for ckpt in f]
                checkpoints = [self.extract_global_step(ckpt) for ckpt in checkpoints[1:]]
                checkpoints = list(filter(lambda gs: gs < current_global_step, checkpoints))
                if len(checkpoints) > self.keep_eval_results_max_epoch:
                    checkpoint_to_delete = checkpoints[-self.keep_eval_results_max_epoch]
                    tf.logging.info("Deleting %s results at the step %d", self.mode, checkpoint_to_delete)
                    tfrecord_filespec = os.path.join(self.writer.get_logdir(),
                                                     "eval_result_step{:09d}_*.tfrecord".format(checkpoint_to_delete))
                    alignment_filespec = os.path.join(self.writer.get_logdir(),
                                                      "alignment_eval_result_step{:09d}_*.png".format(
                                                          checkpoint_to_delete))
                    mel_filespec = os.path.join(self.writer.get_logdir(),
                                                "mel_eval_result_step{:09d}_*.png".format(checkpoint_to_delete))
                    for pathname in tf.gfile.Glob([tfrecord_filespec, alignment_filespec, mel_filespec]):
                        file_io.delete_file(pathname)

    def extract_global_step(self, checkpoint_str):
        return int(self.checkpoint_pattern.match(checkpoint_str)[1])
