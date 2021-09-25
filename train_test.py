# GCT634 (2021) HW1
#
# Sep-26-2021: updated version
#
# Jiyun Park
#

import sys
import os
import numpy as np
import librosa
from sklearn.linear_model import SGDClassifier
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from texttable import Texttable

from feature_summary import *


def extract_accuracy(classifier, train_X, train_Y, valid_X, valid_Y):
    classifier.fit(train_X, train_Y)

    # validation
    valid_Y_hat = classifier.predict(valid_X)
    accuracy = np.sum((valid_Y_hat == valid_Y)) / 300.0 * 100.0

    return classifier, accuracy


def _train_by_SGD_classifier(train_X, train_Y, valid_X, valid_Y, alpha):
    classifier = SGDClassifier(
        verbose=0,
        loss="hinge",
        alpha=alpha,
        max_iter=1000,
        penalty="l2",
        random_state=0,
    )
    return extract_accuracy(classifier, train_X, train_Y, valid_X, valid_Y)


def _train_by_KNN_classifier(train_X, train_Y, valid_X, valid_Y, algorithm, weight):
    classifier = KNeighborsClassifier(
        n_neighbors=10, algorithm=algorithm, weights=weight
    )
    return extract_accuracy(classifier, train_X, train_Y, valid_X, valid_Y)


def _train_by_NuSVC_classifier(train_X, train_Y, valid_X, valid_Y, nu, kernel):
    classifier = NuSVC(nu=nu, kernel=kernel)
    return extract_accuracy(classifier, train_X, train_Y, valid_X, valid_Y)


def _train_by_MLP_classifier(
    train_X, train_Y, valid_X, valid_Y, activation, solver, learning_rate, alpha
):
    classifier = MLPClassifier(
        activation=activation,
        solver=solver,
        max_iter=5000,
        learning_rate=learning_rate,
        alpha=alpha,
    )
    return extract_accuracy(classifier, train_X, train_Y, valid_X, valid_Y)


def _train_by_GMM_classifier(train_X, train_Y, valid_X, valid_Y):
    classifier = GaussianProcessClassifier()
    return extract_accuracy(classifier, train_X, train_Y, valid_X, valid_Y)


def train_model(train_X, train_Y, valid_X, valid_Y, model: str, table: Texttable):
    classifier, accuracy = None, None
    classifiers = []
    valid_acc = []

    # Choose a classifier (here, linear SVM)
    if model == "SGD":  # 96%
        alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        for alpha in alphas:
            classifier, accuracy = _train_by_SGD_classifier(
                train_X, train_Y, valid_X, valid_Y, alpha
            )
            classifiers.append(classifier)
            valid_acc.append(accuracy)
            table.add_row([str(classifier), f"alpha={alpha}", "X", accuracy])
    elif model == "K-NN":  # 89.667% (weight: distance)
        algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
        weights = ["uniform", "distance"]
        for algorithm in algorithms:
            for weight in weights:
                classifier, accuracy = _train_by_KNN_classifier(
                    train_X, train_Y, valid_X, valid_Y, algorithm, weight
                )
                classifiers.append(classifier)
                valid_acc.append(accuracy)
                table.add_row(
                    [
                        str(classifier),
                        f"algorithm={algorithm}, weight={weight}",
                        "X",
                        accuracy,
                    ]
                )
    elif model == "NuSVC":  # 97.333% (nu: 0.1, kernel: rbf)
        nus = [0.1, 0.3, 0.5, 0.9]
        kernels = ["linear", "poly", "rbf", "sigmoid"]
        for nu in nus:
            for kernel in kernels:
                classifier, accuracy = _train_by_NuSVC_classifier(
                    train_X, train_Y, valid_X, valid_Y, nu, kernel
                )
                classifiers.append(classifier)
                valid_acc.append(accuracy)
                table.add_row(
                    [str(classifier), f"nu={nu}, kernel={kernel}", "X", accuracy]
                )
    elif (
        model == "MLP"
    ):  # [MAX] 97.667% (activation: relu, alpha: 0.1, solver: adam, learning_rate: constant)
        alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        activations = ["identity", "logistic", "tanh", "relu"]
        solvers = ["lbfgs", "sgd", "adam"]
        learning_rates = ["constant", "invscaling", "adaptive"]
        for activation in activations:
            for solver in solvers:
                for learning_rate in learning_rates:
                    for alpha in alphas:
                        classifier, accuracy = _train_by_MLP_classifier(
                            train_X, train_Y, valid_X, valid_Y, activation, solver, learning_rate, alpha
                        )
                        classifiers.append(classifier)
                        valid_acc.append(accuracy)
                        table.add_row(
                            [
                                str(classifier),
                                f"activation={activation}, alpha={alpha}, solver={solver}, learning_rate={learning_rate}",
                                "X",
                                accuracy,
                            ]
                        )
    elif model == "GMM":  # 94.333%
        classifier, accuracy = _train_by_GMM_classifier(
            train_X, train_Y, valid_X, valid_Y
        )
        classifiers.append(classifier)
        valid_acc.append(accuracy)
        table.add_row([str(classifier), "", "X", accuracy])

    final_model = classifiers[np.argmax(valid_acc)]
    valid_Y_hat = final_model.predict(valid_X)
    accuracy = np.sum((valid_Y_hat == valid_Y)) / 300.0 * 100.0
    table.add_row([str(final_model), "", "O", accuracy])
    return final_model, accuracy, table


if __name__ == "__main__":
    # load data
    train_X = mean_mfcc("train")
    valid_X = mean_mfcc("valid")

    # label generation
    cls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_Y = np.repeat(cls, 110)
    valid_Y = np.repeat(cls, 30)

    # feature normalization
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)

    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X / (train_X_std + 1e-5)

    model = []
    valid_acc = []
    algorithms = ["SGD", "K-NN", "NuSVC", "MLP", "GMM"]
    for algorithm in algorithms:
        table = Texttable(max_width=300)
        table.header(["classifier", "hyper param", "is final?", "accuracy"])
        final_model, accuracy, table = train_model(
            train_X, train_Y, valid_X, valid_Y, algorithm, table
        )
        model.append(final_model)
        valid_acc.append(accuracy)
        table.set_deco(Texttable.HEADER)
        print(f"[{algorithm}]")
        print(table.draw() + "\n")

    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    valid_Y_hat = final_model.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y)) / 300.0 * 100.0
    print(f"final validation accuracy is {accuracy} with {final_model}")