# Musical Instrument Recognition

This project is based on the lecture 'Musical Application of Machine Learning(GCT634, 2021 fall) by Prof. Juhan Nam of KAIST.

The course aims to learn machine learning with applications to in the music and audio domains. Specificially, it handles various tasks in the topics of music and audio classification, automatic music transcription, source separation, sound synthesis, and music generation. 
This project focuses on the audio data representation & extraction, and classification with traditional machine learning models.

---

## Abstract

The goal of the project is, given datasets with short audio(`.wav`) file of ten different musical instruments, classifying each dataset into valid musical instrument family. Mainly, the task is divided into a task of audio feature extraction, task of summarizing features, and task of classification.
Over a dataset of 1100 tones from 10 musical instruments, features such as MFCC, RMS, Zero crossing rate, spectral centroid, spectral bandwidth have been used for extracting feature information.
On top of these extracted features, 5 classification methods are implemented and tested (SGD, K-NN, NuSVC, MLP, and GMM). All classifiers have their hyper parameters adjusted to have over 90% accuracy. The highest accuracy was 98% by NuSVC model, followed by 97.67% by MLP classifier model.

## Environment Setup & Dataset

```shell
$ conda create -n music-inst python=3.8 jupyter jupyterlab matplotlib
$ conda activate music-inst
(music-inst) $ python -m ipykernel install --user --name music-inst
(music-inst) $ conda install -y -c conda-forge librosa
```

We use a subset of the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) which is a large collection of musical instrument tones from the Google Magenta project. The subset has 10 classes of different musical instruments, including bass, brass, flute, guitar, keyboard, mallet, organ, reed, string and vocal. For our expriment, it is split into training, validation and test sets. For each class, the training set has 110 audio samples and the validation set have 30 audio samples. You can download the subset [here](https://drive.google.com/drive/folders/1-HSAKbbY1ohTW4KylvmB25o0aHrJurbk?usp=sharing). 

Make `./dataset/` directory, and download the dataset to `./dataset/`.
Once you downloaded the dataset, make sure that you have the following files and folders.  

```shell
$ cd dataset
$ ls
train train_list.txt valid valid_list.txt
$ cd ..      # go back to your home folder for next steps
```

## Feature Extraction

```shell
$ python feature_extraction.py
extracting features from ./dataset/train_list.txt
100%|███████████████████████████████████████████████████████████| 1100/1100 [02:12<00:00,  8.33it/s]
extracting features from ./dataset/valid_list.txt
100%|███████████████████████████████████████████████████████████| 300/300 [00:38<00:00,  7.70it/s]
```

`feature_extraction.py` loads audio files and extracts spectral features and stores them in `./mfcc` directory.

In order to extract feature that characterizes each musical instrument, the following aspects can be considered.

- loudness
- continuity of a sound (ex. piano striking vs. flute vibration)
- unique timbre

To capture those aspects from audio, extracting frame-level audio features are needed which summarize a single frame of time-frequency representations into a single value or a vector. In this project, the following features are extracted through `librosa`:

- **Root-Mean-Square(RMS)**: computes the amplitude envelope of waveforms. Useful for volume analysis or continuity of a sound and also used in music structure analysis.
- **Zero crossing rate**: the rate of change of sign of waveform changes within a particular frame.
- **Mel-Frequency Cepstral Coefficient (MFCC)**: extracts the spectrum envelop from an audio frame, invariant to pitch information.
  - Computation steps: 

      <img src="./img/mfcc_step.png" width="600" align="center">

    1. From audio file, compute STFT(matrix `S`) with `n_fft=1024, hop_length=512, win_length=1024`.
    2. Conduct power to spectrum by `D = np.abs(S) ** 2`
    3. Apply Mel filter with `n_mels=128` bins.
    4. Conduct log compression(`log_mel_S`) by `librosa.power_to_db()`
    5. Extract MFCC by DCT with 13 bins to reduce the dimensionality as low frequency range has the most of information. (`librosa.feature.mfcc(S=log_mel_S, n_mfcc=13)`)
    6. Add delta and second delta of MFCC (`librosa.feature.delta()`)
- **Spectral statistics**
  - **centroid**: The spectral centroid is the center of gravity of the magnitude spectrum of STFT ('Center of mass'). It captures the trend in magnitude response, associated with the brightness of sounds. 
  - **bandwidth**: The spectral bandwidth captures whether the spectrum is wide or narrow using standard deviation.
  - **contrast**: The spectral contrast computes the energy contrast estimated by comparing the mean energy in the peak energy to that of valley energy.


## Feature Summary

```shell
$ python feature_summary.py
```
<img src="./img/mfcc.png" width="400" align="center">

> In order to just plot MFCC visually in 13 dimensions in the image above, only the MFCC feature was extracted for convenience. The original code extracts much more features other than MFCC.

For summarizing features, mean pooling in time domain is conducted.

## Train & Test

```shell
$ python train_test.py
[SGD]
                classifier                   hyper param    accuracy
====================================================================
SGDClassifier(random_state=0)                alpha=0.0001   95.333
SGDClassifier(alpha=0.001, random_state=0)   alpha=0.001    95
SGDClassifier(alpha=0.01, random_state=0)    alpha=0.01     93.667
SGDClassifier(alpha=0.1, random_state=0)     alpha=0.1      91.667
SGDClassifier(alpha=1, random_state=0)       alpha=1        83.667
SGDClassifier(alpha=10, random_state=0)      alpha=10       10

[K-NN]
                classifier                                                        hyper param                            accuracy
==================================================================================================================================
KNeighborsClassifier(n_neighbors=10)                                              algorithm=auto, weight=uniform         93.333
KNeighborsClassifier(n_neighbors=10, weights='distance')                          algorithm=auto, weight=distance        94.333
KNeighborsClassifier(algorithm='ball_tree', n_neighbors=10)                       algorithm=ball_tree, weight=uniform    93.333
KNeighborsClassifier(algorithm='ball_tree', n_neighbors=10, weights='distance')   algorithm=ball_tree, weight=distance   94.333
KNeighborsClassifier(algorithm='kd_tree', n_neighbors=10)                         algorithm=kd_tree, weight=uniform      93.333
KNeighborsClassifier(algorithm='kd_tree', n_neighbors=10, weights='distance')     algorithm=kd_tree, weight=distance     94.333
KNeighborsClassifier(algorithm='brute', n_neighbors=10)                           algorithm=brute, weight=uniform        93.333
KNeighborsClassifier(algorithm='brute', n_neighbors=10, weights='distance')       algorithm=brute, weight=distance       94.333

[NuSVC]
                classifier        hyper param              accuracy
=====================================================================
NuSVC(kernel='linear', nu=0.1)    nu=0.1, kernel=linear    95.667
NuSVC(kernel='poly', nu=0.1)      nu=0.1, kernel=poly      97.333
NuSVC(nu=0.1)                     nu=0.1, kernel=rbf       98
...

[MLP]
                classifier                               hyper param                                          accuracy
===========================================================================================================================
MLPClassifier(alpha=0.01, learning_rate='constant')      alpha=0.01, solver=adam, learning_rate=constant      97.333
MLPClassifier(alpha=0.001, learning_rate='invscaling')   alpha=0.001, solver=adam, learning_rate=invscaling   97.667
MLPClassifier(alpha=0.01, learning_rate='invscaling')    alpha=0.01, solver=adam, learning_rate=invscaling    97
...

[GMM]
        classifier            hyper param   accuracy
=======================================================
GaussianProcessClassifier()                 97
``` 

`train_test.py` runs whole training process of 5 classifier models, and displays out the each accuracy with certain parameters.

5 different classification models are applied to train and test musical instrument classification. The result of accuracy of training is summarized in the table below.

classifier | hyper parameter | highest accuracy
--- | --- | ---
SGD | `alpha=0.001` | 95.333%
K-NN | `weight='distance'` | 94.333%
**NuSVC** | `nu=0.1, kernel='rbf'` | **98%**
MLP | `activation='relu', alpha=0.1, solver='adam', learning_rate='invscaling'` | 97.667%
GPC | `kernel='1.0 * RBF(1.0)', optimizer='fmin_l_bfgs_b'` (default) | 97%

### Experiments with parameters of different classifier models

1. [`sklearn.linear_model.SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html?highlight=sgd#sklearn.linear_model.SGDClassifier)

    Linear classifier with stochastic gradient descent (GSD) training. 
    Tested parameters:

    Parameter | Options 
    --- | ---
    alpha | `[0.0001, 0.001, 0.01, 0.1, 1, 10]`
    loss | `["hinge", "log", "squared_hinge", "modified_huber", "squared_loss", "huber"]`

    - The higher the alpha value, the accuracy tended to get lower for most cases.
    - For the loss function, `"hinge", "log", "squared_hinge"` get the same maximum accuracy above 95%, wherewas regression loss function didn't get the score. (`"squared_loss"`: 54.67%, `"huber"`: 81.67%)

2. [`sklearn.neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

    Tested parameters:

    Parameter | Options
    --- | ---
    algorithm | `["auto", "ball_tree", "kd_tree", "brute"]`
    weights | `["uniform", "distance"]`

    - The accuracy didn't change by the type of algorithms.
    - For the weight function, `"distance"` got slightly better accuracy(94.33%) than the default `"uniform"` function(93.33%). The `"distance"` function has greater influence to closer neighbors of a query point than neighbors further away. 


3. [`sklearn.svm.NuSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC)

    Nu-Supprt Vector classification.
    Tested parameters:

    Parameter | Options 
    --- | ---
    nu | `[0.1, 0.3, 0.5, 0.9]`
    kernel | `["linear", "poly", "rbf", "sigmoid"]`

    - The lower the `nu` value, the higher the accuracy.
    - Overall, `"rbf"` kernel (which is default) had the highest accuracy of 98.0% (nu=0.1), `"sigmoid"` had the least accuracy of 58.33% (nu=0.1)
    - SVC with rbf kernel is non-linear SVC classification.

4. [`sklearn.neural_network.MLPClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
    
    Multi-layer perceptron classifier, based on neural network. 
    Tested parameters:

    Parameter | Options
    --- | ---
    activation | `["identity", "logistic", "tanh", "relu"]`
    solver | `["lbfgs", "sgd", "adam"]`
    alpha | `[0.0001, 0.001, 0.01, 0.1, 1, 10]`
    learning rate | `["constant", "invscaling", "adaptive"]`

    - Among the various options for different parameter, the combination of `activation='relu', alpha=0.1, solver='adam', learning_rate='invscaling'` offered the highest accuracy of 97.67%.

5. [`sklearn.gaussian_process.GaussianProcessClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier)
    
    Gaussian process classification (GPC) based on Laplace approximation.
    Only the default option of each parameter has been tested in this training:

    Parameter | Options (default)
    --- | ---
    kernel | `'1.0 * RBF(1.0)'`
    optimizer | `'fmin_l_bfgs_b'`

    Without any adjusting paramter, the accurary was fairly high for Gaussian Process Classifier.


## Discussion & Insights
- Extracting spectral flatness does not affect the training when there's no white noise in the datasets.
- Chroma mainly extracts tonal characteristics and removes timbre information so it doesn't fit into musical instrument classification problem, especially with 12 scale based western instruments.
- Even if additional feature extraction does not increase the maximum accuracy of classification, the accuracy can be generally improved among most of models if the features are sufficiently extracted to reflect the characteristics of each instrument.
- In K-NN classifier, weight function of 'distance' had better accuracy over 'uniform'. This is mostly because since the audio dataset has similar pattern with monophonic instrument sound of 4 seconds, each value had close distance from each other.

---

## References
- Baseline Code from: https://github.com/juhannam/gct634-ai613-2021/tree/main/hw1
- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing, 10(5), 293-302.
- Agostini, G., Longari, M., & Pollastri, E. (2003). Musical instrument timbres classification with spectral features. EURASIP Journal on Advances in Signal Processing, 2003(1), 1-10.
