# ==============================================================================
# Copyright (c) 2018-2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import tensorflow as tf


def learning_rate_decay_factory(hparams):
    def decay_fn(global_step):
        global_step = global_step * hparams.learning_rate_step_factor
        if not hparams.decay_learning_rate:
            return tf.convert_to_tensor(hparams.initial_learning_rate)
        if hparams.learning_rate_decay_method == "exponential":
            return tf.train.exponential_decay(learning_rate=hparams.initial_learning_rate,
                                              global_step=global_step,
                                              decay_steps=hparams.learning_rate_decay_steps,
                                              decay_rate=hparams.learning_rate_decay_rate)
        elif hparams.learning_rate_decay_method == "exponential_bounded":
            lr = tf.train.exponential_decay(learning_rate=hparams.initial_learning_rate,
                                            global_step=global_step,
                                            decay_steps=hparams.learning_rate_decay_steps,
                                            decay_rate=hparams.learning_rate_decay_rate)
            return tf.maximum(lr, hparams.min_learning_rate_if_bounded)
        elif hparams.learning_rate_decay_method == "piecewise_constant":
            return tf.train.piecewise_constant(x=global_step,
                                               boundaries=hparams.learning_rate_decay_boundaries,
                                               values=hparams.learning_rate_decay_values)
        elif hparams.learning_rate_decay_method == "exponential_legacy":
            warmup_steps = 4000.0
            global_step = tf.to_float(global_step + 1)
            return hparams.initial_learning_rate * warmup_steps ** 0.5 * tf.minimum(global_step * warmup_steps ** -1.5,
                                                                                    global_step ** -0.5)
        else:
            raise ValueError(f"Unknown learning rate decay method: {hparams.learning_rate_decay_method}")

    return decay_fn
