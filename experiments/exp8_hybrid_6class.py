import numpy as np
np.random.seed(42)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from common import (
    RESULTS_DIR,
    TEST_SIZE,
    SPLIT_RANDOM_STATE,
    ensure_results_dir,
    load_subject_split,
    metric_summary,
    plot_confusion_matrix,
    save_json,
)
from emg_features import extract_handcrafted_features
from hidden_temporal_readout import (
    extract_hidden_spike_features,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


N_HIDDEN = 80
TIME_BINS = 5
ENCODER_MODE = "envelope"
FEATURE_CHUNK_SIZE = 256


def extract_features_with_progress(X, num_classes, split_name):
    chunks = []
    total = len(X)
    num_chunks = (total + FEATURE_CHUNK_SIZE - 1) // FEATURE_CHUNK_SIZE

    for chunk_idx, start in enumerate(range(0, total, FEATURE_CHUNK_SIZE), start=1):
        end = min(start + FEATURE_CHUNK_SIZE, total)
        print(
            f"Extracting {split_name} hidden spike features: "
            f"chunk {chunk_idx}/{num_chunks} ({start}:{end})"
        )
        chunk_features = extract_hidden_spike_features(
            X[start:end],
            n_hidden=N_HIDDEN,
            n_outputs=num_classes,
            time_bins=TIME_BINS,
            encoder_mode=ENCODER_MODE,
        )
        chunks.append(chunk_features)

    return np.vstack(chunks)


def load_or_compute_feature_cache(X, num_classes, cache_path, split_name):
    if os.path.exists(cache_path):
        print(f"Loading cached {split_name} hidden spike features from {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            return data["features"]

    print(f"No cached {split_name} features found. Computing from scratch...")
    features = extract_features_with_progress(X, num_classes, split_name)
    np.savez_compressed(cache_path, features=features)
    print(f"Saved {split_name} feature cache to {cache_path}")
    return features


def main():
    print("=== Experiment 8: Hybrid SNN + handcrafted features (6-class) ===")
    ensure_results_dir()

    split = load_subject_split("full_6")
    train_cache_path = os.path.join(RESULTS_DIR, "exp8_hybrid_6class_train_features.npz")
    test_cache_path = os.path.join(RESULTS_DIR, "exp8_hybrid_6class_test_features.npz")
    print(
        f"Train samples: {len(split['y_train'])}, "
        f"Test samples: {len(split['y_test'])}"
    )

    X_train_hidden = load_or_compute_feature_cache(
        split["X_train"],
        split["num_classes"],
        train_cache_path,
        "train",
    )
    print("Extracting train handcrafted features...")
    X_train_handcrafted = extract_handcrafted_features(split["X_train"])

    X_test_hidden = load_or_compute_feature_cache(
        split["X_test"],
        split["num_classes"],
        test_cache_path,
        "test",
    )
    print("Extracting test handcrafted features...")
    X_test_handcrafted = extract_handcrafted_features(split["X_test"])

    X_train_combined = np.hstack([X_train_hidden, X_train_handcrafted])
    X_test_combined = np.hstack([X_test_hidden, X_test_handcrafted])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    print("Training hybrid logistic classifier...")
    clf.fit(X_train_scaled, split["y_train"])

    print("Evaluating hybrid classifier...")
    y_pred = clf.predict(X_test_scaled)

    metrics = metric_summary(split["y_test"], y_pred, split["class_names"])
    json_path = os.path.join(RESULTS_DIR, "exp8_hybrid_6class.json")
    cm_path = os.path.join(RESULTS_DIR, "exp8_hybrid_6class_cm.png")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        split["class_names"],
        cm_path,
        "Experiment 8 Confusion Matrix",
    )
    save_json(
        json_path,
        {
            "experiment": "exp8_hybrid_6class",
            "label_mode": "full_6",
            "feature_mode": "hybrid",
            "readout_mode": "logistic_regression",
            "n_hidden": N_HIDDEN,
            "time_bins": TIME_BINS,
            "encoder_mode": ENCODER_MODE,
            "test_size": TEST_SIZE,
            "split_random_state": SPLIT_RANDOM_STATE,
            "train_samples": len(split["y_train"]),
            "test_samples": len(split["y_test"]),
            "class_names": split["class_names"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "train_feature_cache": train_cache_path,
            "test_feature_cache": test_cache_path,
            "confusion_matrix_png": cm_path,
        },
    )

    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    print(f"Final macro F1: {metrics['macro_f1']:.4f}")
    print("Saved results to results/exp8_hybrid_6class.json")


if __name__ == "__main__":
    main()
