# ==============================================================================
# Copyright (c) 2018-2019, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

import numpy as np
from collections import namedtuple
import tensorflow as tf


class AlignmentError(namedtuple("AlignmentError", ["value", "level"])):
    pass


OK = AlignmentError("OK", 0)
INCOMPLETE = AlignmentError("INCOMPLETE", 5)
DISCONTINUOUS = AlignmentError("DISCONTINUOUS", 10)
OVERESTIMATE = AlignmentError("OVERESTIMATE", 21)
UNDERESTIMATE = AlignmentError("UNDERESTIMATE", 22)


class AlignmentErrorDetector:
    def __init__(self, alignment, threshold, min_duration, max_duration):
        self.alignment = alignment
        self._threshold = threshold
        self._min_duration = min_duration
        self._max_duration = max_duration

    @property
    def input_len(self):
        return self.alignment.shape[0]

    @property
    def output_len(self):
        return self.alignment.shape[1]

    def check_all(self):
        return list(filter(lambda v: v is not OK,
                           [self.continuous(), self.completeness()]))

    def contain_errors(self):
        return len(self.check_all()) > 0

    def continuous(self):
        scores = np.sort(self.alignment, axis=0)
        indices = np.argsort(self.alignment, axis=0)
        soft_topk_attended_idx = np.sum((scores * indices), axis=0)

        def check_adjacent():
            for at, atp1 in zip(soft_topk_attended_idx[:-1], soft_topk_attended_idx[1:]):
                diff = atp1 - at
                yield -self._threshold / 2 <= diff <= self._threshold

        is_continuous = all(list(check_adjacent()))
        return OK if is_continuous else DISCONTINUOUS

    def completeness(self):
        last = self.alignment[:, -1]
        last_attended = np.argmax(last)
        is_complete = self.input_len - last_attended <= self._threshold
        return OK if is_complete else INCOMPLETE

    def overestimated_duration(self):
        duration = np.sum(self.alignment, axis=-1)
        overestimate = np.any(duration > self._max_duration)
        return OVERESTIMATE if overestimate else OK

    def underestimated_duration(self):
        duration = np.sum(self.alignment, axis=-1)
        underestimate = np.any(duration < self._min_duration)
        return UNDERESTIMATE if underestimate else OK


def detect_alignment_error(alignment):
    def func(xx):
        # FixMe
        return 0.0
        errors = [float(AlignmentErrorDetector(x,
                                               threshold=4.0,
                                               min_duration=0.0,
                                               max_duration=30.0).contain_errors()) for x in xx.numpy()]
        return sum(errors) / float(len(errors))

    return tf.py_function(func, [alignment], tf.float32)
