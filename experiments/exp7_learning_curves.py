import numpy as np
np.random.seed(42)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt

from common import (
    RESULTS_DIR,
    TEST_SIZE,
    SPLIT_RANDOM_STATE,
    build_lr_pipeline,
    ensure_results_dir,
    load_subject_split,
    metric_summary,
    rms_mav_features,
    save_json,
)


TRAIN_SIZES = [250, 500, 1000, 2000, 3000]


def main():
    print("=== Experiment 7: Learning curves for handcrafted LR (6-class) ===")
    ensure_results_dir()

    split = load_subject_split("full_6")
    X_train_feat = rms_mav_features(split["X_train"])
    X_test_feat = rms_mav_features(split["X_test"])

    rng = np.random.default_rng(42)
    curve_rows = []

    for train_size in TRAIN_SIZES:
        sample_size = min(train_size, len(split["y_train"]))
        sample_idx = rng.choice(len(split["y_train"]), size=sample_size, replace=False)
        model = build_lr_pipeline()
        model.fit(X_train_feat[sample_idx], split["y_train"][sample_idx])
        y_pred = model.predict(X_test_feat)
        metrics = metric_summary(split["y_test"], y_pred, split["class_names"])
        curve_rows.append(
            {
                "train_size": sample_size,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
        print(
            f"Train size {sample_size}: "
            f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
        )

    plot_path = os.path.join(RESULTS_DIR, "exp7_learning_curves.png")
    json_path = os.path.join(RESULTS_DIR, "exp7_learning_curves.json")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [row["train_size"] for row in curve_rows],
        [row["accuracy"] for row in curve_rows],
        marker="o",
        label="Accuracy",
    )
    ax.plot(
        [row["train_size"] for row in curve_rows],
        [row["macro_f1"] for row in curve_rows],
        marker="s",
        label="Macro F1",
    )
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Score")
    ax.set_title("Experiment 7 Learning Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    save_json(
        json_path,
        {
            "experiment": "exp7_learning_curves",
            "label_mode": "full_6",
            "features": "RMS + MAV per channel",
            "test_size": TEST_SIZE,
            "split_random_state": SPLIT_RANDOM_STATE,
            "train_sizes": TRAIN_SIZES,
            "results": curve_rows,
            "plot_path": plot_path,
        },
    )

    print(f"Final accuracy: {curve_rows[-1]['accuracy']:.4f}")
    print(f"Final macro F1: {curve_rows[-1]['macro_f1']:.4f}")
    print("Saved results to results/exp7_learning_curves.json")


if __name__ == "__main__":
    main()
