"""synthesize waveform
Usage: predict_mel.py [options]

Options:
    --source-data-root=<dir>        Directory contains preprocessed features.
    --target-data-root=<dir>           Directory contains preprocessed features.
    --checkpoint-dir=<dir>          Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>              Hyper parameters. [default: ].
    --hparam-json-file=<path>       JSON file contains hyper parameters.
    --checkpoint=<path>             Restore model from checkpoint path if given.
    --selected-list-dir=<dir>       Directory contains test.lst.validated, train.lst.validated, and val.lst.validated
    --output-dir=<path>             Output directory.
    --selected-list-filename=<name> Selected list file name [default: test.lst.validated]
    -h, --help                      Show this help message and exit
"""

from docopt import docopt
import tensorflow as tf
from tensorflow.python.keras import backend
import os
from typing import List
import numpy as np
from collections import namedtuple
from datasets.dataset_factory import dataset_factory
from core.models import model_factory
from hparams import hparams, hparams_debug_string
from utils.tfrecord import int64_feature, bytes_feature, write_tfrecord
from utils.audio import Audio
import core.metrics as ssnt_metrics


class PredictedMel(
    namedtuple("PredictedMel",
               ["id", "key", "predicted_mel", "predicted_mel_postnet", "predicted_mel_width", "predicted_target_length",
                "ground_truth_mel", "alignment", "alignment2", "alignment3", "alignment4", "alignment5", "alignment6",
                "attention2_gate_activation", "source",
                "text", "accent_type", "all_fields"])):
    pass


def predict(hparams,
            model_dir, checkpoint_path, output_dir,
            test_source_files, test_target_files):
    if hparams.half_precision:
        backend.set_floatx(tf.float16.name)
        backend.set_epsilon(1e-4)

    audio = Audio(hparams)

    def predict_input_fn():
        source = tf.data.TFRecordDataset(list(test_source_files))
        target = tf.data.TFRecordDataset(list(test_target_files))
        dataset = dataset_factory(source, target, hparams)
        batched = dataset.prepare_and_zip().group_by_batch(
            batch_size=1).move_mel_to_source()
        return batched.dataset

    estimator = model_factory(hparams, model_dir, None)

    predictions = map(
        lambda p: PredictedMel(p["id"], p["key"], p["mel"], p.get("mel_postnet"), p["mel"].shape[1], p["mel"].shape[0],
                               p["ground_truth_mel"],
                               p["alignment"], p.get("alignment2"), p.get("alignment3"), p.get("alignment4"),
                               p.get("alignment5"), p.get("alignment6"), p.get("attention2_gate_activation"),
                               p["source"], p["text"], p.get("accent_type"), p),
        estimator.predict(predict_input_fn, checkpoint_path=checkpoint_path))

    for v in predictions:
        key = v.key.decode('utf-8')
        mel_filename = f"{key}.{hparams.predicted_mel_extension}"
        mel_filepath = os.path.join(output_dir, mel_filename)
        ground_truth_mel = v.ground_truth_mel.astype(np.float32)
        predicted_mel = v.predicted_mel.astype(np.float32)
        mel_denormalized = audio.denormalize_mel(predicted_mel)

        linear_spec = audio.logmelspc_to_linearspc(mel_denormalized)
        wav = audio.griffin_lim(linear_spec)
        audio.save_wav(wav, os.path.join(output_dir, f"{key}.wav"))

        assert mel_denormalized.shape[1] == hparams.num_mels
        mel_denormalized.tofile(mel_filepath, format='<f4')
        text = v.text.decode("utf-8")
        plot_filename = f"{key}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        alignments = [x.astype(np.float32) for x in [
            v.alignment, v.alignment2, v.alignment3, v.alignment4, v.alignment5, v.alignment6] if x is not None]

        if hparams.model == "SSNTModel":
            ssnt_metrics.save_alignment_and_log_probs([v.alignment],
                                                      [v.all_fields["log_emit_and_shift_probs"]],
                                                      [v.all_fields["log_output_prob"]],
                                                      [None],
                                                      text, v.key, 0,
                                                      os.path.join(output_dir, f"{key}_probs.png"))
            ssnt_metrics.write_prediction_result(v.id, key, text,
                                                 v.all_fields["log_emit_and_shift_probs"],
                                                 v.all_fields["log_output_prob"],
                                                 os.path.join(output_dir, f"{key}_probs.tfrecord"))
        plot_predictions(alignments, ground_truth_mel, predicted_mel, text, v.key, plot_filepath)
        prediction_filename = f"{key}.tfrecord"
        prediction_filepath = os.path.join(output_dir, prediction_filename)
        write_prediction_result(v.id, key, alignments, mel_denormalized, audio.denormalize_mel(ground_truth_mel),
                                text, v.source, prediction_filepath)


def write_prediction_result(id_: int, key: str, alignments: List[np.ndarray], mel: np.ndarray,
                            ground_truth_mel: np.ndarray,
                            text: str, source: np.ndarray, filename: str):
    example = tf.train.Example(features=tf.train.Features(feature={
        'id': int64_feature([id_]),
        'key': bytes_feature([key.encode('utf-8')]),
        'mel': bytes_feature([mel.tostring()]),
        'mel_length': int64_feature([mel.shape[0]]),
        'mel_width': int64_feature([mel.shape[1]]),
        'ground_truth_mel': bytes_feature([ground_truth_mel.tostring()]),
        'ground_truth_mel_length': int64_feature([ground_truth_mel.shape[0]]),
        'alignment': bytes_feature([alignment.tostring() for alignment in alignments]),
        'text': bytes_feature([text.encode('utf-8')]),
        'source': bytes_feature([source.tostring()]),
        'source_length': int64_feature([source.shape[0]]),
    }))
    write_tfrecord(example, filename)


def plot_predictions(alignments, mel, mel_predicted, text, _id, filename):
    from matplotlib import pylab as plt
    num_alignment = len(alignments)
    num_rows = num_alignment + 3
    fig = plt.figure(figsize=(12, num_rows * 3))

    for i, alignment in enumerate(alignments):
        ax = fig.add_subplot(num_rows, 1, i + 1)
        im = ax.imshow(
            alignment,
            aspect='auto',
            origin='lower',
            interpolation='none',
            cmap='jet')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Encoder timestep')
        ax.set_title("layer {}".format(i + 1))
    fig.subplots_adjust(wspace=0.4, hspace=0.6)

    ax = fig.add_subplot(num_rows, 1, num_alignment + 1)
    im = ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax = fig.add_subplot(num_rows, 1, num_alignment + 2)
    im = ax.imshow(mel_predicted[:mel.shape[0], :].T,
                   origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)

    fig.suptitle(f"record ID: {_id}\ninput text: {str(text)}")
    fig.savefig(filename, format='png')
    plt.close()


def load_key_list(filename, in_dir):
    path = os.path.join(in_dir, filename)
    with open(path, mode="r", encoding="utf-8") as f:
        for l in f:
            yield l.rstrip("\n")


def main():
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    source_data_root = args["--source-data-root"]
    mel_data_root = args["--target-data-root"]
    selected_list_dir = args["--selected-list-dir"]
    output_dir = args["--output-dir"]
    selected_list_filename = args["--selected-list-filename"] or "test.lst.validated"

    tf.logging.set_verbosity(tf.logging.INFO)

    if args["--hparam-json-file"]:
        with open(args["--hparam-json-file"]) as f:
            json = "".join(f.readlines())
            hparams.parse_json(json)

    hparams.parse(args["--hparams"])
    tf.logging.info(hparams_debug_string())

    tf.logging.info(f"A selected list file to use: {os.path.join(selected_list_dir, selected_list_filename)}")

    test_list = list(load_key_list(selected_list_filename, selected_list_dir))

    test_source_files = [os.path.join(source_data_root, f"{key}.{hparams.source_file_extension}") for key in
                         test_list]
    test_target_files = [os.path.join(mel_data_root, f"{key}.{hparams.target_file_extension}") for key in
                         test_list]

    predict(hparams,
            checkpoint_dir,
            checkpoint_path,
            output_dir,
            test_source_files,
            test_target_files)


if __name__ == '__main__':
    main()
