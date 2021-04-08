import librosa
import numpy as np
from scipy.io.wavfile import write as write_wav


class Audio:

    def __init__(self, hparams):
        self.hparams = hparams
        self._mel_basis = self._build_mel_basis()
        self.average_mel_level_db = np.array(hparams.average_mel_level_db, dtype=np.float32)
        self.stddev_mel_level_db = np.array(hparams.stddev_mel_level_db, dtype=np.float32)

    def _build_mel_basis(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        return librosa.filters.mel(self.hparams.sample_rate, n_fft, n_mels=self.hparams.num_mels)

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.hparams.sample_rate)[0]

    def save_wav(self, wav, path):
        wav = wav * 32767 / max(0.01, np.max(np.abs(wav)))
        write_wav(path, self.hparams.sample_rate, wav.astype(np.int16))

    def trim(self, wav):
        unused_trimed, index = librosa.effects.trim(wav, top_db=self.hparams.trim_top_db,
                                                    frame_length=self.hparams.trim_frame_length,
                                                    hop_length=self.hparams.trim_hop_length)
        num_sil_samples = int(
            self.hparams.num_silent_frames * self.hparams.frame_shift_ms * self.hparams.sample_rate / 1000)
        start_idx = max(index[0] - num_sil_samples, 0)
        stop_idx = min(index[1] + num_sil_samples, len(wav))
        trimed = wav[start_idx:stop_idx]
        return trimed

    def silence_frames(self, wav, trim_frame_length, trim_hop_length):
        unused_trimed, index = librosa.effects.trim(wav, top_db=self.hparams.trim_top_db,
                                                    frame_length=self.hparams.trim_frame_length,
                                                    hop_length=self.hparams.trim_hop_length)
        num_start_frames = int((index[0] - trim_frame_length + trim_hop_length) / trim_hop_length)
        num_start_frames = max(num_start_frames - self.hparams.num_silent_frames, 0)
        num_stop_frames = int(((len(wav) - index[1]) - trim_frame_length + trim_hop_length) / trim_hop_length)
        num_stop_frames = min(num_stop_frames + self.hparams.num_silent_frames, len(wav))
        return num_start_frames, num_stop_frames

    def melspectrogram(self, y, espnet_compatible=False):
        D = self._stft(y)
        if espnet_compatible:
            S = self._amp_to_log_espnet(self._linear_to_mel(np.abs(D)))
        else:
            S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hparams.ref_level_db
        return S

    def denormalize_mel(self, S):
        return (S * self.stddev_mel_level_db) + self.average_mel_level_db

    def logmelspc_to_linearspc(self, lmspc):
        n_fft = (self.hparams.num_freq - 1) * 2
        n_mels = self.hparams.num_mels
        assert lmspc.shape[1] == n_mels
        mspc = self._db_to_amp(lmspc + self.hparams.ref_level_db)
        mel_basis = librosa.filters.mel(self.hparams.sample_rate, n_fft,
                                        n_mels=self.hparams.num_mels,
                                        fmin=self.hparams.mel_fmin,
                                        fmax=self.hparams.mel_fmax)
        inv_mel_basis = np.linalg.pinv(mel_basis)
        spc = np.maximum(1e-10, np.dot(inv_mel_basis, mspc.T).T)
        return spc

    def griffin_lim(self, spc):
        n_fft = (self.hparams.num_freq - 1) * 2
        assert spc.shape[1] == n_fft // 2 + 1
        cspc = np.abs(spc).astype(np.complex).T
        angles = np.exp(2j * np.pi * np.random.rand(*cspc.shape))
        y = self._istft(cspc * angles)
        for i in range(self.hparams.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(cspc * angles)

        return y

    def _linear_to_mel(self, spectrogram):
        return np.dot(self._mel_basis, spectrogram)

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _amp_to_log_espnet(self, x):
        return np.log10(np.maximum(1e-10, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _stft_parameters(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        hop_length = int(self.hparams.frame_shift_ms / 1000 * self.hparams.sample_rate)
        win_length = int(self.hparams.frame_length_ms / 1000 * self.hparams.sample_rate)
        return n_fft, hop_length, win_length

    def _istft(self, stfts):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.istft(stfts, hop_length, win_length)

