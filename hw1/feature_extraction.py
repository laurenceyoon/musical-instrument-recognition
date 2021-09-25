# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#
from tqdm import tqdm
import os
import numpy as np
import librosa

data_path = './dataset/'
mfcc_path = './mfcc/'

SAMPLE_RATE = 22050
WIN_LENGTH = 1024
HOP_LENGTH = 512
N_FFT = 1024
MEL_BINS = 128  # n_mels, used for mel filtering
MFCC_DIM = 13  # used for mfcc (dct)

def extract_mfcc(dataset='train'):
    with open(f"{data_path}{dataset}_list.txt", "r") as f:
        print(f"extracting mfcc from {f.name}")
    
    for file_name in tqdm(f.readlines()):
        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        ##### Method 1
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)
        
        ##### Method 2 
        """
        # STFT
        S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024)

        # power spectrum
        D = np.abs(S)**2

        # mel spectrogram (512 --> 40)
        mel_basis = librosa.filters.mel(sr, 1024, n_mels=40)
        mel_S = np.dot(mel_basis, D)

        #log compression
        log_mel_S = librosa.power_to_db(mel_S)

        # mfcc (DCT)
        mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=13)
        mfcc = mfcc.astype(np.float32)    # to save the memory (64 to 32 bits)
        """

        # save mfcc as a file
        file_name = file_name.replace('.wav','.npy')
        save_file = mfcc_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc)


if __name__ == '__main__':
    extract_mfcc(dataset='train')                 
    extract_mfcc(dataset='valid')                                  

