import tensorflow as tf
import math

hparams = tf.contrib.training.HParams(

    # Audio
    num_mels=80,
    num_mgcs=60,
    num_freq=513,
    sample_rate=22050,
    frame_length_ms=46.4399092971,
    frame_shift_ms=11.6099773243,
    mel_fmin=125,
    mel_fmax=7600,
    ref_level_db=20,
    average_mel_level_db=[0.0],
    stddev_mel_level_db=[0.0],
    silence_mel_level_db=-3.0,

    # Dataset
    dataset="ljspeech.dataset.DatasetSource",
    num_symbols=256,
    source_file_extension="source.tfrecord",
    target_file_extension="target.tfrecord",
    predicted_mel_extension="mfbsp",

    # Model:
    model="SSNTModel",
    half_precision=False,
    outputs_per_step=2,
    n_feed_frame=2,

    ## Embedding
    embedding_dim=256,

    ### accent
    use_accent_type=False,
    accent_type_embedding_dim=32,
    num_accent_type=129,
    accent_type_offset=0x3100,
    accent_type_unknown=0x3180,

    ## RNN
    lstm_impl="tf.nn.rnn_cell.LSTMCell",  # tf.nn.rnn_cell.LSTMCell, tf.contrib.rnn.LSTMBlockCell

    ## Encoder
    # encoder= ZoneoutEncoderV1 | EncoderV2
    encoder="ZoneoutEncoderV1",

    ### Encoder V1
    encoder_prenet_drop_rate=0.5,
    cbhg_out_units=256,
    conv_channels=128,
    max_filter_width=16,
    projection1_out_channels=128,
    projection2_out_channels=128,
    num_highway=4,
    encoder_prenet_out_units=(256, 128),
    use_zoneout_at_encoder=False,
    zoneout_factor_cell=0.1,
    zoneout_factor_output=0.1,

    ### Encoder V2
    encoder_v2_num_conv_layers=3,
    encoder_v2_kernel_size=5,
    encoder_v2_out_units=512,
    encoder_v2_drop_rate=0.5,

    ## Decoder
    decoder_prenet_drop_rate=0.5,
    apply_dropout_on_inference=False,
    decoder_prenet_out_units=(256, 128),
    attention_out_units=256,
    attention_rnn_out_units=(256,),  # Fix other decoders than  to use this params.
    decoder_out_units=256,

    ## SSNT
    ssnt_unconditional_layer_num_units=[256, 256],
    ssnt_conditional_layer_num_units=[256, 256],
    ssnt_num_bijection_layers=1,
    ssnt_sampling_mode="transition",  # transition | joint
    ssnt_alignment_sampling_mode="deterministic",  # deterministic | stochastic
    ssnt_output_sampling_mode="deterministic",  # deterministic | stochastic
    ssnt_input_and_feedback_combination_mode="concat",  # concat | add
    ssnt_output_stddev=1.0,
    ssnt_sigmoid_noise=0.0,
    ssnt_sigmoid_temperature=1.0,
    ssnt_predict_sigmoid_temperature=False,
    ssnt_beam_width=1,
    ssnt_beam_search_score_bias=0.0,
    ssnt_bijection_type="identity",  # maf | identity
    ssnt_maf_bijector_num_hidden_layers=2,
    ssnt_maf_bijector_num_hidden_units=640,
    ssnt_maf_bijector_num_blocks=160,
    ssnt_maf_bijector_activation_function="relu",  # relu, tanh

    ## loss
    spec_loss_type="l1",  # l1 or mse

    # Training:
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,  # deprecated
    gradient_clip_norm=1.0,
    initial_learning_rate=0.002,
    decay_learning_rate=True,
    learning_rate_decay_method="exponential_legacy",
    # exponential, exponential_bounded, piecewise_constant, exponential_legacy
    learning_rate_step_factor=1,
    learning_rate_decay_steps=1000,  # exponential
    learning_rate_decay_rate=0.96,  # exponential
    min_learning_rate_if_bounded=1e-5,  # exponential_bounded
    learning_rate_decay_boundaries=[5000, 10000, 80000],  # piecewise_constant
    learning_rate_decay_values=[3e-4, 1e-4, 5e-5, 1e-5],  # piecewise_constant
    use_l2_regularization=False,
    l2_regularization_weight=1e-7,
    l2_regularization_weight_black_list=["embedding", "bias", "batch_normalization", "output_projection_wrapper/kernel",
                                         "lstm_cell",
                                         "output_and_stop_token_wrapper/dense/", "output_and_stop_token_wrapper/dense_1/",
                                         "stop_token_projection/kernel"],
    save_summary_steps=100,
    save_checkpoints_steps=500,
    keep_checkpoint_max=200,
    keep_checkpoint_every_n_hours=1,  # deprecated
    log_step_count_steps=1,
    alignment_save_steps=10000,
    save_training_time_metrics=False,
    approx_min_target_length=100,
    suffle_buffer_size=64,
    batch_bucket_width=50,
    batch_num_buckets=50,
    interleave_cycle_length_cpu_factor=1.0,
    interleave_cycle_length_min=4,
    interleave_cycle_length_max=16,
    interleave_buffer_output_elements=200,
    interleave_prefetch_input_elements=200,
    prefetch_buffer_size=4,
    use_cache=False,
    cache_file_name="",
    logfile="log.txt",
    record_profile=False,
    profile_steps=50,
    ## Warm starting
    warm_start=False,
    ckpt_to_initialize_from="",
    vars_to_warm_start=[".*"],

    # Eval:
    max_iters=500,
    griffin_lim_iters=60,
    power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim
    num_evaluation_steps=64,
    keep_eval_results_max_epoch=10,
    eval_start_delay_secs=120,
    eval_throttle_secs=600,
    num_epochs_per_evaluation=1,  # deprecated

    # Predict
    renormalize_mel_output=False,
    use_forced_alignment_mode=False,
    use_teacher_alignment=False,

)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
