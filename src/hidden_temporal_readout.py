import json
import os

import numpy as np
from brian2 import *

from snn import build_snn
from windowing import (
    DEFAULT_SNN_DELTA_GAIN,
    DEFAULT_SNN_ENCODER_MODE,
    DEFAULT_SNN_INPUT_GAIN,
    preprocess_snn_window,
)


SIM_TIME = 200 * ms
FEATURE_RANDOM_STATE = 42


def default_prototypes_path(
    results_dir,
    label_mode,
    n_hidden,
    time_bins,
    encoder_mode
):
    return os.path.join(
        results_dir,
        f"hidden_proto_{label_mode}_{encoder_mode}_h{n_hidden}_tb{time_bins}.npz"
    )


def extract_hidden_spike_features(
    X,
    n_hidden,
    n_outputs,
    time_bins=5,
    encoder_mode=DEFAULT_SNN_ENCODER_MODE,
    input_gain=DEFAULT_SNN_INPUT_GAIN,
    delta_gain=DEFAULT_SNN_DELTA_GAIN,
    random_state=FEATURE_RANDOM_STATE
):
    np.random.seed(random_state)
    seed(random_state)

    net, input_g, hidden, output, syn_out, spike_hidden, spike_out = build_snn(
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        encoder_mode=encoder_mode,
        record_spike_trains=(time_bins > 1),
    )

    feature_dim = int(hidden.N) * int(time_bins)
    features = np.zeros((len(X), feature_dim), dtype=float)
    prev_hidden_counts = np.array(spike_hidden.count)
    prev_hidden_spike_total = 0

    for idx, window in enumerate(X):
        sample = np.array(window, dtype=float)

        if sample.ndim == 2 and sample.shape[0] == int(input_g.N):
            sample = sample.T

        T = int(sample.shape[0])
        dt = SIM_TIME / float(T)

        scaled_window = preprocess_snn_window(
            sample,
            input_gain=input_gain,
            delta_gain=delta_gain,
            encoder_mode=encoder_mode,
        )
        input_signal = TimedArray(scaled_window.T, dt=dt, name="input_signal")

        input_g.v = 0
        hidden.v = 0
        output.v = 0

        t_start = float(net.t / second)
        net.run(SIM_TIME, namespace={"input_signal": input_signal})

        current_hidden_counts = np.array(spike_hidden.count)
        hidden_spike_counts = current_hidden_counts - prev_hidden_counts

        if time_bins <= 1:
            features[idx] = hidden_spike_counts
        else:
            sample_features = np.zeros((int(hidden.N), int(time_bins)), dtype=float)
            all_i = np.array(spike_hidden.i)
            all_t = np.array(spike_hidden.t / second)

            sample_i = all_i[prev_hidden_spike_total:]
            sample_t = all_t[prev_hidden_spike_total:]

            if len(sample_i) > 0:
                rel_ms = (sample_t - t_start) * 1000.0
                bins = np.linspace(0.0, float(SIM_TIME / ms), int(time_bins) + 1)

                for neuron_idx in range(int(hidden.N)):
                    neuron_times = rel_ms[sample_i == neuron_idx]
                    if len(neuron_times) == 0:
                        continue
                    hist, _ = np.histogram(neuron_times, bins=bins)
                    sample_features[neuron_idx, :] = hist

            features[idx] = sample_features.reshape(-1)
            prev_hidden_spike_total = len(all_i)

        prev_hidden_counts = current_hidden_counts

    return features


def fit_class_prototypes(features, labels, num_classes):
    prototypes = np.zeros((num_classes, features.shape[1]), dtype=float)

    for cls in range(num_classes):
        rows = features[labels == cls]
        if len(rows) == 0:
            continue
        prototypes[cls] = np.mean(rows, axis=0)

    return prototypes


def predict_with_prototypes(features, prototypes):
    preds = []
    prototype_norms = np.sum(prototypes ** 2, axis=1)

    for row in features:
        dists = np.sum((prototypes - row[np.newaxis, :]) ** 2, axis=1)
        if np.sum(row) == 0.0:
            preds.append(int(np.argmin(prototype_norms)))
        else:
            preds.append(int(np.argmin(dists)))

    return np.array(preds, dtype=int)


def save_prototypes(
    path,
    prototypes,
    label_mode,
    class_names,
    n_hidden,
    time_bins,
    encoder_mode,
    input_gain,
    delta_gain
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    metadata = {
        "label_mode": label_mode,
        "class_names": class_names,
        "n_hidden": int(n_hidden),
        "time_bins": int(time_bins),
        "encoder_mode": encoder_mode,
        "input_gain": float(input_gain),
        "delta_gain": float(delta_gain),
    }
    np.savez(path, prototypes=prototypes, metadata_json=json.dumps(metadata))
    return metadata


def load_prototypes(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing prototype file at {path}. "
            "Run train_snn.py with --readout-mode hidden_temporal_prototype first."
        )

    data = np.load(path, allow_pickle=False)
    prototypes = data["prototypes"]
    metadata = json.loads(str(data["metadata_json"]))
    return prototypes, metadata
