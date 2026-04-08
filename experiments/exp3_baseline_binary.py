import numpy as np
np.random.seed(42)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from common import (
    RESULTS_DIR,
    TEST_SIZE,
    SPLIT_RANDOM_STATE,
    build_lr_pipeline,
    ensure_results_dir,
    load_subject_split,
    metric_summary,
    plot_confusion_matrix,
    save_json,
)
from emg_features import extract_handcrafted_features


def main():
    print("=== Experiment 3: Handcrafted LR baseline (binary) ===")
    ensure_results_dir()

    split = load_subject_split("rest_vs_active")
    X_train_feat = extract_handcrafted_features(split["X_train"])
    X_test_feat = extract_handcrafted_features(split["X_test"])

    model = build_lr_pipeline()
    model.fit(X_train_feat, split["y_train"])
    y_pred = model.predict(X_test_feat)

    metrics = metric_summary(split["y_test"], y_pred, split["class_names"])
    json_path = os.path.join(RESULTS_DIR, "exp3_baseline_binary.json")
    cm_path = os.path.join(RESULTS_DIR, "exp3_baseline_binary_cm.png")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        split["class_names"],
        cm_path,
        "Experiment 3 Confusion Matrix",
    )
    save_json(
        json_path,
        {
            "experiment": "exp3_baseline_binary",
            "label_mode": "rest_vs_active",
            "features": "src.emg_features.extract_handcrafted_features",
            "test_size": TEST_SIZE,
            "split_random_state": SPLIT_RANDOM_STATE,
            "train_samples": len(split["y_train"]),
            "test_samples": len(split["y_test"]),
            "class_names": split["class_names"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "confusion_matrix_png": cm_path,
        },
    )

    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    print(f"Final macro F1: {metrics['macro_f1']:.4f}")
    print("Saved results to results/exp3_baseline_binary.json")


if __name__ == "__main__":
    main()
