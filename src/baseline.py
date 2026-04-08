import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from build_dataset import build_full_dataset, split_dataset_by_subject
from emg_features import extract_handcrafted_features
from experiment_logging import append_experiment_summary
from label_modes import DEFAULT_LABEL_MODE, get_class_names


DEFAULT_RESULTS_DIR = "results"


def run_baseline(label_mode=DEFAULT_LABEL_MODE, results_dir=DEFAULT_RESULTS_DIR):
    print("Loading dataset...")
    X, y, subjects = build_full_dataset(label_mode=label_mode)
    class_names = get_class_names(label_mode)

    print("Extracting handcrafted EMG features...")
    X_feat = extract_handcrafted_features(X)
    print("Feature shape:", X_feat.shape)
    print("Class names:", class_names)

    # Subject-wise split
    train_idx, test_idx = split_dataset_by_subject(
        X_feat,
        y,
        subjects,
        test_size=0.3,
        random_state=42
    )

    X_train, X_test = X_feat[train_idx], X_feat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Train size:", len(y_train))
    print("Test size:", len(y_test))

    # -----------------------------
    # FEATURE SCALING (CRITICAL)
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=10.0,
        solver="lbfgs",
        n_jobs=1
    )

    print("Training Logistic Regression...")
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    cm_path = os.path.join(results_dir, f"baseline_confusion_matrix_{label_mode}.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    results = {
        "label_mode": label_mode,
        "class_names": class_names,
        "feature_mode": "handcrafted",
        "readout_mode": "logistic_regression",
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "confusion_matrix_png": cm_path,
    }

    results_path = os.path.join(results_dir, f"baseline_results_{label_mode}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    results["results_json"] = results_path
    summary_path = append_experiment_summary(
        results,
        results_dir=results_dir,
        script="baseline.py",
        mode="baseline",
    )

    print("\n===== BASELINE RESULTS =====")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Macro F1   : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"\nSaved confusion matrix -> {cm_path}")
    print(f"Saved evaluation results -> {results_path}")
    print(f"Updated experiment summary -> {summary_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the handcrafted EMG baseline.")
    parser.add_argument(
        "--label-mode",
        default=DEFAULT_LABEL_MODE,
        help="Label mode to evaluate."
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for plots and JSON outputs."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(label_mode=args.label_mode, results_dir=args.results_dir)
