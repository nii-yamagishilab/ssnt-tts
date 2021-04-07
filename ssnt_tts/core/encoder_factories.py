# ==============================================================================
# Copyright (c) 2018-2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Yusuke Yasuda (yasuda@nii.ac.jp)
# All rights reserved.
# ==============================================================================
"""  """

from tensorflow.python.keras import backend
from core.modules import ZoneoutEncoderV1, EncoderV2


def encoder_factory(params, is_training, dtype=None):
    dtype = dtype or backend.floatx()
    if params.encoder == "ZoneoutEncoderV1":
        encoder = ZoneoutEncoderV1(is_training,
                                   cbhg_out_units=params.cbhg_out_units,
                                   conv_channels=params.conv_channels,
                                   max_filter_width=params.max_filter_width,
                                   projection1_out_channels=params.projection1_out_channels,
                                   projection2_out_channels=params.projection2_out_channels,
                                   num_highway=params.num_highway,
                                   prenet_out_units=params.encoder_prenet_out_units,
                                   drop_rate=params.encoder_prenet_drop_rate,
                                   use_zoneout=params.use_zoneout_at_encoder,
                                   zoneout_factor_cell=params.zoneout_factor_cell,
                                   zoneout_factor_output=params.zoneout_factor_output,
                                   lstm_impl=params.lstm_impl,
                                   dtype=dtype)
    elif params.encoder == "EncoderV2":
        encoder = EncoderV2(num_conv_layers=params.encoder_v2_num_conv_layers,
                            kernel_size=params.encoder_v2_kernel_size,
                            out_units=params.encoder_v2_out_units,
                            drop_rate=params.encoder_v2_drop_rate,
                            zoneout_factor_cell=params.zoneout_factor_cell,
                            zoneout_factor_output=params.zoneout_factor_output,
                            is_training=is_training,
                            lstm_impl=params.lstm_impl,
                            dtype=dtype)
    else:
        raise ValueError(f"Unknown encoder: {params.encoder}")
    return encoder
