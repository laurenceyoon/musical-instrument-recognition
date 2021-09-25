# GCT634 (2021) HW1
#
# Sep-26-2021: updated version
#
# Jiyun Park
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

data_path = "./dataset/"
mfcc_path = "./mfcc/"

MFCC_DIM = 50


def mean_mfcc(dataset="train"):
    with open(f"{data_path}{dataset}_list.txt", "r") as f:
        if dataset == "train":
            mfcc_mat = np.zeros(shape=(MFCC_DIM, 1100))
        else:
            mfcc_mat = np.zeros(shape=(MFCC_DIM, 300))

        for i, file_name in enumerate(f):
            # load mfcc file
            file_name = file_name.rstrip("\n")
            file_name = file_name.replace(".wav", ".npy")
            mfcc_file = mfcc_path + file_name
            mfcc = np.load(mfcc_file)

            # mean pooling
            temp = np.mean(mfcc, axis=1)
            mfcc_mat[:, i] = np.mean(mfcc, axis=1)

    return mfcc_mat


if __name__ == "__main__":
    train_data = mean_mfcc("train")
    valid_data = mean_mfcc("valid")

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.imshow(train_data, interpolation="nearest", origin="lower", aspect="auto")
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(2, 1, 2)
    plt.imshow(valid_data, interpolation="nearest", origin="lower", aspect="auto")
    plt.colorbar(format="%+2.0f dB")

    plt.show()
