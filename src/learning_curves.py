import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from build_dataset import build_full_dataset, split_dataset_by_subject
from emg_features import extract_handcrafted_features
from eval_snn_hidden_readout import extract_hidden_spike_features
from label_modes import DEFAULT_LABEL_MODE, get_class_names, num_classes_for_mode
from windowing import (
    DEFAULT_SNN_DELTA_GAIN,
    DEFAULT_SNN_ENCODER_MODE,
    DEFAULT_SNN_INPUT_GAIN,
)


DEFAULT_RESULTS_DIR = "results"
DEFAULT_HIDDEN = 10
DEFAULT_TIME_BINS = 5
DEFAULT_FEATURE_MODES = ("handcrafted", "hybrid")
DEFAULT_TRAIN_SIZES = (250, 500, 1000, 2000, 3000)
DEFAULT_TEST_SAMPLES = 1000
SPLIT_RANDOM_STATE = 42


def parse_list_arg(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_int_list_arg(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def choose_indices(total_size, limit, random_state):
    if limit is None or limit >= total_size:
        return np.arange(total_size)

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(total_size, size=limit, replace=False))


def fit_and_score(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }


def build_feature_blocks(
    X_train,
    X_test,
    label_mode,
    n_hidden,
    time_bins,
    encoder_mode,
    input_gain,
    delta_gain,
    feature_modes,
):
    num_classes = num_classes_for_mode(label_mode)
    blocks = {}

    if any(mode in {"handcrafted", "hybrid"} for mode in feature_modes):
        print("Extracting handcrafted EMG features...")
        blocks["handcrafted_train"] = extract_handcrafted_features(X_train)
        blocks["handcrafted_test"] = extract_handcrafted_features(X_test)

    if any(mode in {"hidden", "hybrid"} for mode in feature_modes):
        print("Extracting hidden spike features for train set...")
        blocks["hidden_train"] = extract_hidden_spike_features(
            X_train,
            n_hidden=n_hidden,
            n_outputs=num_classes,
            time_bins=time_bins,
            encoder_mode=encoder_mode,
            input_gain=input_gain,
            delta_gain=delta_gain,
        )
        print("Extracting hidden spike features for test set...")
        blocks["hidden_test"] = extract_hidden_spike_features(
            X_test,
            n_hidden=n_hidden,
            n_outputs=num_classes,
            time_bins=time_bins,
            encoder_mode=encoder_mode,
            input_gain=input_gain,
            delta_gain=delta_gain,
        )

    return blocks


def select_mode_features(blocks, feature_mode, train_idx):
    if feature_mode == "handcrafted":
        return blocks["handcrafted_train"][train_idx], blocks["handcrafted_test"]
    if feature_mode == "hidden":
        return blocks["hidden_train"][train_idx], blocks["hidden_test"]
    if feature_mode == "hybrid":
        X_train = np.concatenate(
            [blocks["hidden_train"][train_idx], blocks["handcrafted_train"][train_idx]],
            axis=1,
        )
        X_test = np.concatenate(
            [blocks["hidden_test"], blocks["handcrafted_test"]],
            axis=1,
        )
        return X_train, X_test
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def generate_learning_curves(
    label_mode=DEFAULT_LABEL_MODE,
    feature_modes=DEFAULT_FEATURE_MODES,
    train_sizes=DEFAULT_TRAIN_SIZES,
    test_samples=DEFAULT_TEST_SAMPLES,
    n_hidden=DEFAULT_HIDDEN,
    time_bins=DEFAULT_TIME_BINS,
    encoder_mode=DEFAULT_SNN_ENCODER_MODE,
    input_gain=DEFAULT_SNN_INPUT_GAIN,
    delta_gain=DEFAULT_SNN_DELTA_GAIN,
    results_dir=DEFAULT_RESULTS_DIR,
):
    feature_modes = tuple(feature_modes)
    train_sizes = sorted(set(int(size) for size in train_sizes))

    print("Loading dataset...")
    X, y, subjects = build_full_dataset(label_mode=label_mode)
    class_names = get_class_names(label_mode)

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

    max_train_size = min(max(train_sizes), len(X_train_all))
    train_pool_idx = choose_indices(len(X_train_all), max_train_size, random_state=42)
    test_pool_idx = choose_indices(len(X_test_all), test_samples, random_state=43)

    X_train_pool = X_train_all[train_pool_idx]
    y_train_pool = y_train_all[train_pool_idx]
    X_test = X_test_all[test_pool_idx]
    y_test = y_test_all[test_pool_idx]

    rng = np.random.default_rng(44)
    nested_order = rng.permutation(len(X_train_pool))

    print("Label mode:", label_mode)
    print("Class names:", class_names)
    print("Feature modes:", list(feature_modes))
    print("Train sizes:", train_sizes)
    print("Test samples:", len(X_test))
    print("Hidden units:", n_hidden)
    print("Time bins:", time_bins)
    print("Encoder mode:", encoder_mode)

    blocks = build_feature_blocks(
        X_train_pool,
        X_test,
        label_mode=label_mode,
        n_hidden=n_hidden,
        time_bins=time_bins,
        encoder_mode=encoder_mode,
        input_gain=input_gain,
        delta_gain=delta_gain,
        feature_modes=feature_modes,
    )

    results = {
        "label_mode": label_mode,
        "class_names": class_names,
        "feature_modes": list(feature_modes),
        "train_sizes": train_sizes,
        "test_samples": int(len(X_test)),
        "n_hidden": int(n_hidden),
        "time_bins": int(time_bins),
        "encoder_mode": encoder_mode,
        "input_gain": float(input_gain),
        "delta_gain": float(delta_gain),
        "curves": {},
    }

    for feature_mode in feature_modes:
        print(f"Building curve for feature mode: {feature_mode}")
        points = []
        for size in train_sizes:
            use_size = min(int(size), len(X_train_pool))
            subset_idx = nested_order[:use_size]
            X_train_feat, X_test_feat = select_mode_features(blocks, feature_mode, subset_idx)
            metrics = fit_and_score(X_train_feat, y_train_pool[subset_idx], X_test_feat, y_test)
            point = {
                "train_size": int(use_size),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
            points.append(point)
            print(
                f"  train_size={use_size}: "
                f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
            )

        results["curves"][feature_mode] = points

    if "hidden_test" in blocks:
        results["test_zero_hidden_rate"] = float(np.mean(np.sum(blocks["hidden_test"], axis=1) == 0.0))
    if "hidden_train" in blocks:
        results["train_zero_hidden_rate"] = float(np.mean(np.sum(blocks["hidden_train"], axis=1) == 0.0))

    os.makedirs(results_dir, exist_ok=True)
    suffix = f"{label_mode}_{encoder_mode}_h{n_hidden}_tb{time_bins}"
    json_path = os.path.join(results_dir, f"learning_curves_{suffix}.json")
    csv_path = os.path.join(results_dir, f"learning_curves_{suffix}.csv")
    png_path = os.path.join(results_dir, f"learning_curves_{suffix}.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("feature_mode,train_size,accuracy,macro_f1\n")
        for feature_mode, points in results["curves"].items():
            for point in points:
                f.write(
                    f"{feature_mode},{point['train_size']},"
                    f"{point['accuracy']:.6f},{point['macro_f1']:.6f}\n"
                )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
    ]

    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        for feature_mode, points in results["curves"].items():
            x = [point["train_size"] for point in points]
            y_vals = [point[metric_key] for point in points]
            ax.plot(x, y_vals, marker="o", linewidth=2, label=feature_mode)
        ax.set_xlabel("Training samples")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.suptitle(f"Learning Curves: {label_mode}")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print("\n===== LEARNING CURVES COMPLETE =====")
    print(f"Saved plot -> {png_path}")
    print(f"Saved CSV -> {csv_path}")
    print(f"Saved JSON -> {json_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate learning curves for handcrafted and hybrid multiclass readouts."
    )
    parser.add_argument("--label-mode", default=DEFAULT_LABEL_MODE)
    parser.add_argument(
        "--feature-modes",
        default="handcrafted,hybrid",
        help="Comma-separated feature modes, e.g. handcrafted,hybrid",
    )
    parser.add_argument(
        "--train-sizes",
        default="250,500,1000,2000,3000",
        help="Comma-separated training sizes to evaluate.",
    )
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--time-bins", type=int, default=DEFAULT_TIME_BINS)
    parser.add_argument("--encoder-mode", default=DEFAULT_SNN_ENCODER_MODE)
    parser.add_argument("--input-gain", type=float, default=DEFAULT_SNN_INPUT_GAIN)
    parser.add_argument("--delta-gain", type=float, default=DEFAULT_SNN_DELTA_GAIN)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_learning_curves(
        label_mode=args.label_mode,
        feature_modes=parse_list_arg(args.feature_modes),
        train_sizes=parse_int_list_arg(args.train_sizes),
        test_samples=args.test_samples,
        n_hidden=args.hidden,
        time_bins=args.time_bins,
        encoder_mode=args.encoder_mode,
        input_gain=args.input_gain,
        delta_gain=args.delta_gain,
        results_dir=args.results_dir,
    )
