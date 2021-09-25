# Musical Instrument Recognition

Jiyun Park

GCT634: Musical Application of Machine Learning (Prof. Juhan Nam)

Submission Date: 2021/09/26

---

## Abstract

The goal of the project is, given datasets with short audio(`.wav`) file of ten different musical instruments, classifying each dataset into valid musical instrument family. Mainly, the task is divided into a task of audio feature extraction, task of summarizing features, and task of classification.
Over a dataset of 1100 tones from 10 musical instruments, features such as MFCC, RMS, Zero crossing rate, spectral centroid, spectral bandwidth have been used for extracting feature information.
On top of these extracted features, 5 classification methods are implemented and tested (SGD, K-NN, NuSVC, MLP, and GMM). All classifiers have their hyper parameters adjusted to have over 90% accuracy. The highest accuracy was 98% by NuSVC model, followed by 97.67% by MLP classifier model.


## Feature Extraction

```python
$ python feature_extraction.py
```

`feature_extraction.py` loads audio files and extracts spectral features and stores them in `./mfcc` directory.

In order to extract feature that characterizes each musical instrument, the following aspects can be considered.

- loudness
- continuity of a sound (ex. piano striking vs. flute vibration)
- unique timbre


## Feature Summary

For summarizing features, mean pooling in time domain is conducted.

## Train & Test

```python
$ python train_test.py
```

`train_test.py` runs whole traning process of 5 classifier models, and displays out the each accuracy with certain parameters.

5 different classification models are applied to train and test musical instrument classification. The result of accuracy of training is summarized in the table below.

classifier | hyper parameter | highest accuracy
--- | --- | ---
SGD | `alpha=0.001` | 95.333%
K-NN | `weight='distance'` | 94.333%
**NuSVC** | `nu=0.1, kernel='rbf'` | **98%**
MLP | `alpha=0.001, solver='adam', learning_rate='invscaling'` | 97.667%
GMM | `kernel='1.0 * RBF(1.0)', optimizer='fmin_l_bfgs_b'` (default) | 97%

## Discussion & Insights