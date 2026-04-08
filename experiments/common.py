import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_dataset import build_full_dataset, split_dataset_by_subject
from label_modes import get_class_names, num_classes_for_mode


TEST_SIZE = 0.3
SPLIT_RANDOM_STATE = 42
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2)


def plot_confusion_matrix(cm, class_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def load_subject_split(label_mode):
    X, y, subjects = build_full_dataset(label_mode=label_mode)
    train_idx, test_idx = split_dataset_by_subject(
        X,
        y,
        subjects,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )
    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
        "train_subjects": subjects[train_idx],
        "test_subjects": subjects[test_idx],
        "class_names": get_class_names(label_mode),
        "num_classes": num_classes_for_mode(label_mode),
    }


def rms_mav_features(X):
    X = np.asarray(X, dtype=float)
    rms = np.sqrt(np.mean(np.square(X), axis=1))
    mav = np.mean(np.abs(X), axis=1)
    return np.concatenate([rms, mav], axis=1)


def build_lr_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def metric_summary(y_true, y_pred, class_names):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm,
    }
