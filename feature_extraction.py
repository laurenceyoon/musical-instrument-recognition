# GCT634 (2021) HW1
#
# Sep-26-2021: updated version
#
# Jiyun Park
#

from tqdm import tqdm
import os
import numpy as np
import librosa

data_path = "./dataset/"
mfcc_path = "./mfcc/"

SAMPLE_RATE = 22050
WIN_LENGTH = 1024
HOP_LENGTH = 512
N_FFT = 1024
MEL_BINS = 128  # n_mels, used for mel filtering
MFCC_DIM = 13  # used for mfcc (dct)


def extract_mfcc(dataset="train"):
    with open(f"{data_path}{dataset}_list.txt", "r") as f:
        print(f"extracting mfcc from {f.name}")

        for file_name in tqdm(f.readlines()):
            # load audio file
            file_name = file_name.rstrip("\n")
            file_path = data_path + file_name
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # STFT
            S = librosa.core.stft(
                y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
            )
            # power spectrum
            D = np.abs(S) ** 2

            # mel spectrogram
            mel_basis = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=MEL_BINS)
            mel_S = np.dot(mel_basis, D)

            # log compression
            log_mel_S = librosa.power_to_db(mel_S)

            # MFCC (DCT)
            mfcc = librosa.feature.mfcc(sr=sr, S=log_mel_S, n_mfcc=MFCC_DIM)
            mfcc = mfcc.astype(np.float32)  # to save the memory (64 to 32 bits)

            # Add delta & second delta of mfcc
            delta_mfcc = librosa.feature.delta(mfcc)
            delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)

            # rms & zero crossing rate feature
            f_rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH, frame_length=WIN_LENGTH)
            f_zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y=y, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH
            )

            # spectral statistics features
            centroid = librosa.feature.spectral_centroid(
                S=mel_S, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH
            )
            # flatness = librosa.feature.spectral_flatness(S=mel_S, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
            # ^-- commented out because all datasets are not white noise.
            bandwidth = librosa.feature.spectral_bandwidth(
                S=mel_S, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
            )
            contrast = librosa.feature.spectral_contrast(
                S=mel_S, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
            )

            # concatenate all features
            features = np.concatenate(
                [
                    mfcc,
                    delta_mfcc,
                    delta_delta_mfcc,
                    f_rms,
                    f_zero_crossing_rate,
                    centroid,
                    bandwidth,
                    contrast,
                ],
                axis=0,
            )

            # save mfcc features as a file
            file_name = file_name.replace(".wav", ".npy")
            save_file = mfcc_path + file_name

            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, features)


if __name__ == "__main__":
    extract_mfcc(dataset="train")
    extract_mfcc(dataset="valid")
