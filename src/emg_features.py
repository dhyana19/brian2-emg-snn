import numpy as np


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=0))


def mav(x):
    return np.mean(np.abs(x), axis=0)


def waveform_length(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)


def zero_crossings(x):
    prod = x[:-1] * x[1:]
    return np.sum(prod < 0, axis=0)


def extract_handcrafted_features(X):
    features = []
    for window in X:
        feat = np.concatenate([
            rms(window),
            mav(window),
            waveform_length(window),
            zero_crossings(window),
        ])
        features.append(feat)
    return np.array(features)
