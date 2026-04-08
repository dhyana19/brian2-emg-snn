import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from build_dataset import build_full_dataset, split_dataset_by_subject
from emg_features import extract_handcrafted_features
from experiment_logging import append_experiment_summary
from label_modes import DEFAULT_LABEL_MODE, get_class_names, num_classes_for_mode
from snn import build_snn
from windowing import (
    DEFAULT_SNN_DELTA_GAIN,
    DEFAULT_SNN_ENCODER_MODE,
    DEFAULT_SNN_INPUT_GAIN,
    preprocess_snn_window,
)


SIM_TIME = 200 * ms
DEFAULT_HIDDEN = 10
DEFAULT_RESULTS_DIR = "results"
SPLIT_RANDOM_STATE = 42
FEATURE_RANDOM_STATE = 42


def format_class_counts(y, num_classes):
    counts = np.bincount(y.astype(int), minlength=num_classes)
    return {int(i): int(counts[i]) for i in range(num_classes)}


def choose_indices(total_size, limit, random_state):
    if limit is None or limit >= total_size:
        return np.arange(total_size)

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(total_size, size=limit, replace=False))


def extract_hidden_spike_features(
    X,
    n_hidden,
    n_outputs,
    time_bins=1,
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

    features = np.zeros((len(X), int(hidden.N) * int(time_bins)), dtype=float)
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


def eval_snn_hidden_readout(
    label_mode=DEFAULT_LABEL_MODE,
    n_hidden=DEFAULT_HIDDEN,
    train_samples=None,
    test_samples=None,
    time_bins=1,
    feature_mode="hidden",
    encoder_mode=DEFAULT_SNN_ENCODER_MODE,
    input_gain=DEFAULT_SNN_INPUT_GAIN,
    delta_gain=DEFAULT_SNN_DELTA_GAIN,
    results_dir=DEFAULT_RESULTS_DIR,
    artifact_suffix=None,
):
    print("Loading dataset...")
    X, y, subjects = build_full_dataset(label_mode=label_mode)

    train_idx, test_idx = split_dataset_by_subject(
        X,
        y,
        subjects,
        test_size=0.3,
        random_state=SPLIT_RANDOM_STATE,
    )

    X_train_all = X[train_idx]
    y_train_all = y[train_idx]
    X_test_all = X[test_idx]
    y_test_all = y[test_idx]

    num_classes = num_classes_for_mode(label_mode)
    class_names = get_class_names(label_mode)

    train_pick = choose_indices(len(X_train_all), train_samples, random_state=42)
    test_pick = choose_indices(len(X_test_all), test_samples, random_state=43)

    X_train = X_train_all[train_pick]
    y_train = y_train_all[train_pick]
    X_test = X_test_all[test_pick]
    y_test = y_test_all[test_pick]

    print("Label mode:", label_mode)
    print("Class names:", class_names)
    print("Hidden units:", n_hidden)
    print("Time bins:", time_bins)
    print("Feature mode:", feature_mode)
    print("Encoder mode:", encoder_mode)
    print("Input gain:", input_gain)
    print("Delta gain:", delta_gain)
    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("Train class counts:", format_class_counts(y_train, num_classes))
    print("Test class counts:", format_class_counts(y_test, num_classes))

    X_train_feat = None
    X_test_feat = None
    train_zero_rate = None
    test_zero_rate = None

    if feature_mode in {"hidden", "hybrid"}:
        print("Extracting hidden spike features for train set...")
        X_train_hidden = extract_hidden_spike_features(
            X_train,
            n_hidden=n_hidden,
            n_outputs=num_classes,
            time_bins=time_bins,
            encoder_mode=encoder_mode,
            input_gain=input_gain,
            delta_gain=delta_gain,
            random_state=FEATURE_RANDOM_STATE,
        )

        print("Extracting hidden spike features for test set...")
        X_test_hidden = extract_hidden_spike_features(
            X_test,
            n_hidden=n_hidden,
            n_outputs=num_classes,
            time_bins=time_bins,
            encoder_mode=encoder_mode,
            input_gain=input_gain,
            delta_gain=delta_gain,
            random_state=FEATURE_RANDOM_STATE,
        )

        train_zero_rate = float(np.mean(np.sum(X_train_hidden, axis=1) == 0.0))
        test_zero_rate = float(np.mean(np.sum(X_test_hidden, axis=1) == 0.0))

    if feature_mode in {"handcrafted", "hybrid"}:
        print("Extracting handcrafted EMG features...")
        X_train_hand = extract_handcrafted_features(X_train)
        X_test_hand = extract_handcrafted_features(X_test)

    if feature_mode == "hidden":
        X_train_feat = X_train_hidden
        X_test_feat = X_test_hidden
    elif feature_mode == "handcrafted":
        X_train_feat = X_train_hand
        X_test_feat = X_test_hand
    elif feature_mode == "hybrid":
        X_train_feat = np.concatenate([X_train_hidden, X_train_hand], axis=1)
        X_test_feat = np.concatenate([X_test_hidden, X_test_hand], axis=1)
    else:
        raise ValueError(
            f"Unknown feature_mode '{feature_mode}'. "
            "Valid options: hidden, handcrafted, hybrid."
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    print("Training logistic readout on hidden spike features...")
    clf.fit(X_train_scaled, y_train)

    print("Evaluating hidden-feature readout...")
    y_pred = clf.predict(X_test_scaled)

    labels = np.arange(num_classes)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    clf_report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    os.makedirs(results_dir, exist_ok=True)
    if artifact_suffix is None:
        artifact_suffix = (
            f"_{label_mode}_{feature_mode}_readout_{encoder_mode}_"
            f"h{n_hidden}_tb{time_bins}"
        )

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    out_cm = os.path.join(results_dir, f"confusion_matrix{artifact_suffix}.png")
    plt.savefig(out_cm, bbox_inches="tight")
    plt.close(fig)

    hidden_spike_means = []
    for cls in labels:
        class_rows = X_test_feat[y_test == cls]
        if len(class_rows) == 0:
            hidden_spike_means.append([0.0 for _ in range(n_hidden)])
        else:
            hidden_spike_means.append(np.mean(class_rows, axis=0).astype(float).tolist())

    top_features = []
    for cls_idx, cls_name in enumerate(class_names):
        weights = clf.coef_[cls_idx]
        best_idx = np.argsort(np.abs(weights))[-5:][::-1].tolist()
        top_features.append({
            "class_name": cls_name,
            "top_hidden_indices": [int(i) for i in best_idx],
            "top_hidden_weights": [float(weights[i]) for i in best_idx],
        })

    results = {
        "label_mode": label_mode,
        "class_names": class_names,
        "n_hidden": int(n_hidden),
        "time_bins": int(time_bins),
        "feature_mode": feature_mode,
        "encoder_mode": encoder_mode,
        "input_gain": float(input_gain),
        "delta_gain": float(delta_gain),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_class_counts": format_class_counts(y_train, num_classes),
        "test_class_counts": format_class_counts(y_test, num_classes),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "train_zero_hidden_rate": train_zero_rate,
        "test_zero_hidden_rate": test_zero_rate,
        "mean_hidden_spikes_per_class": hidden_spike_means,
        "top_readout_features": top_features,
    }

    out_json = os.path.join(results_dir, f"results{artifact_suffix}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    results["results_json"] = out_json
    results["confusion_matrix_png"] = out_cm
    summary_path = append_experiment_summary(
        results,
        results_dir=results_dir,
        script="eval_snn_hidden_readout.py",
        mode="hidden_readout",
    )

    print("\n===== SNN HIDDEN-READOUT RESULTS =====")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {f1:.4f}")
    print(f"Time bins    : {time_bins}")
    if train_zero_rate is not None and test_zero_rate is not None:
        print(f"Zero hidden spikes (train): {train_zero_rate:.2%}")
        print(f"Zero hidden spikes (test) : {test_zero_rate:.2%}")
    print("Confusion Matrix:")
    print(cm)
    print(f"\nSaved confusion matrix -> {out_cm}")
    print(f"Saved evaluation results -> {out_json}")
    print(f"Updated experiment summary -> {summary_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an SNN hidden-spike feature readout."
    )
    parser.add_argument(
        "--label-mode",
        default=DEFAULT_LABEL_MODE,
        help="Label mode to evaluate.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=DEFAULT_HIDDEN,
        help="Number of hidden neurons in the SNN feature extractor.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Optional number of training windows to subsample.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=200,
        help="Optional number of held-out test windows to subsample.",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=1,
        help="Number of temporal bins per hidden neuron. Use 1 for total spike counts.",
    )
    parser.add_argument(
        "--feature-mode",
        default="hidden",
        help="Feature mode: hidden, handcrafted, or hybrid.",
    )
    parser.add_argument(
        "--encoder-mode",
        default=DEFAULT_SNN_ENCODER_MODE,
        help="Input encoder mode: envelope or envelope_delta.",
    )
    parser.add_argument(
        "--input-gain",
        type=float,
        default=DEFAULT_SNN_INPUT_GAIN,
        help="Gain applied to the rectified/smoothed envelope channels.",
    )
    parser.add_argument(
        "--delta-gain",
        type=float,
        default=DEFAULT_SNN_DELTA_GAIN,
        help="Gain applied to delta channels when using envelope_delta mode.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for plots and JSON outputs.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Optional suffix for saved artifact names.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_snn_hidden_readout(
        label_mode=args.label_mode,
        n_hidden=args.hidden,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        time_bins=args.time_bins,
        feature_mode=args.feature_mode,
        encoder_mode=args.encoder_mode,
        input_gain=args.input_gain,
        delta_gain=args.delta_gain,
        results_dir=args.results_dir,
        artifact_suffix=args.artifact_suffix,
    )
