# SSNT-TTS

An implementation for SSNT-TTS.

Please refer to the following papers for details.
- [Initial investigation of an encoder-decoder end-to-end TTS framework using marginalization of monotonic hard latent alignments](https://arxiv.org/abs/1908.11535)
- [Effect of choice of probability distribution, randomness, and search methods for alignment modeling in sequence-to-sequence text-to-speech synthesis using hard alignment](https://arxiv.org/abs/1910.12383)

## Install

### Python 3.7

- docopt=0.6.2
- hypothesis=6.8.3
- inflect=5.3.0
- librosa=0.8.0
- matplotlib=3.3.4
- pyspark=2.4.0
- tensorflow=1.14.0
- unidecode=1.2.0
- [ssnt-tts-rust](https://github.com/nii-yamagishilab/ssnt-tts-rust)

### Others

- [JRE8 (java runtime required by pyspark)](https://adoptopenjdk.net/)
- [Cargo (Rust compiler and dependency manager required by ssnt-tts-rust)](https://rustup.rs/)

### ssnt-tts-rust

```
git clone git@github.com:nii-yamagishilab/ssnt-tts-rust.git
cd ssnt-tts-rust/ssnt_tts_c 
cargo build --release
cd ..
cd ssnt-tts-tensorflow 
python setup.py install
```

## Preprocess

An example for [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

Source date
```
python preprocess_ljspeech.py --source-only <path to LJSpech directory> <path to output directory>
```

Target data
```
python preprocess_ljspeech.py --target-only --hparams=sample_rate=22050,num_freq=513,frame_shift_ms=11.6099773243,frame_length_ms=46.4399092971,mel_fmin=125,mel_fmax=7600 <path to LJSpech directory> <path to output directory>
```

## Training

An example for logistic condition.

```
python train.py --source-data-root=<source data directory> --target-data-root=<target data directory> --checkpoint-dir=<checkpoint directory> --selected-list-dir=`pwd`/../ljspeech_selected_list --hparams=model="SSNTModel",encoder="EncoderV2",encoder_v2_out_units=512,accent_type_embedding_dim=64,embedding_dim=448,initial_learning_rate=0.0001,decay_learning_rate=True,learning_rate_decay_method="exponential_bounded",decoder_prenet_drop_rate=0.5,encoder_prenet_drop_rate=0.5,outputs_per_step=2,max_iters=500,use_zoneout_at_encoder=True,save_checkpoints_steps=2000,keep_checkpoint_max=100,num_symbols=256,eval_throttle_secs=600,eval_start_delay_secs=120,ssnt_unconditional_layer_num_units=[512],ssnt_conditional_layer_num_units=[512],batch_size=100,ssnt_bijection_type="identity",ssnt_num_bijection_layers=1,ssnt_sigmoid_noise=0.0,num_evaluation_steps=32,ssnt_alignment_sampling_mode="deterministic",ssnt_output_stddev=0.6,gradient_clip_norm=10.0,interleave_cycle_length_max=4,ssnt_maf_bijector_num_hidden_units=640,ssnt_maf_bijector_num_hidden_layers=1,ssnt_maf_bijector_activation_function="tanh",ssnt_sigmoid_temperature=1.0,logfile=log.txt --hparam-json-file=<path to target data directory>/hparams.json
```


## Prediction

An example for deterministic inference with beam search.

```
python  predict_mel.py --output-dir=<path to output directory> --source-data-root=<source data directory> --target-data-root=<target data directory> --checkpoint-dir=<checkpoint directory> --selected-list-dir=`pwd`/../ljspeech_selected_list --hparams=model="SSNTModel",encoder="EncoderV2",encoder_v2_out_units=512,accent_type_embedding_dim=64,embedding_dim=448,initial_learning_rate=0.0001,decay_learning_rate=True,learning_rate_decay_method="exponential_bounded",decoder_prenet_drop_rate=0.5,encoder_prenet_drop_rate=0.5,outputs_per_step=2,max_iters=500,use_zoneout_at_encoder=True,save_checkpoints_steps=2000,keep_checkpoint_max=100,num_symbols=256,eval_throttle_secs=600,eval_start_delay_secs=120,ssnt_unconditional_layer_num_units=[512],ssnt_conditional_layer_num_units=[512],batch_size=100,ssnt_bijection_type="identity",ssnt_num_bijection_layers=1,ssnt_sigmoid_noise=0.0,num_evaluation_steps=32,ssnt_alignment_sampling_mode="deterministic",ssnt_output_stddev=0.6,gradient_clip_norm=10.0,interleave_cycle_length_max=4,ssnt_maf_bijector_num_hidden_units=640,ssnt_maf_bijector_num_hidden_layers=1,ssnt_maf_bijector_activation_function="tanh",ssnt_sigmoid_temperature=1.0,logfile=log.txt,ssnt_beam_width=10 --hparam-json-file=<path to target data directory>/hparams.json
```


## Licence

BSD 3-Clause License

Copyright (c) 2021, Yusuke Yasuda at Yamagishi Laboratory, National Institute of Informatics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.