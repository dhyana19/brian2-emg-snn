import argparse
import json
import os

from train_snn import train_snn
from eval_snn import eval_snn
from label_modes import DEFAULT_LABEL_MODE


DEFAULT_HIDDEN_SIZES = [10, 20, 40, 80]
DEFAULT_RESULTS_DIR = "results"


def run_hidden_size_sweep(
    hidden_sizes,
    epochs,
    n_train,
    test_samples,
    label_mode=DEFAULT_LABEL_MODE,
    results_dir=DEFAULT_RESULTS_DIR
):
    os.makedirs(results_dir, exist_ok=True)
    summary_rows = []

    for hidden in hidden_sizes:
        print("\n" + "=" * 60)
        print(f"Running hidden-size experiment: {hidden} ({label_mode})")
        print("=" * 60)

        train_info = train_snn(
            label_mode=label_mode,
            n_hidden=hidden,
            n_train=n_train,
            epochs=epochs,
            results_dir=results_dir,
        )

        eval_info = eval_snn(
            label_mode=label_mode,
            n_hidden=hidden,
            test_samples=test_samples,
            results_dir=results_dir,
            weights_path=train_info["weights_path"],
            artifact_suffix=f"_{label_mode}_h{hidden}",
        )

        row = {
            "label_mode": label_mode,
            "n_hidden": int(hidden),
            "epochs": int(epochs),
            "n_train": int(n_train),
            "test_samples": None if test_samples is None else int(test_samples),
            "checkpoint_path": train_info["weights_path"],
            "accuracy": float(eval_info["accuracy"]),
            "macro_f1": float(eval_info["macro_f1"]),
            "zero_output_rate": float(eval_info["zero_output_rate"]),
            "prediction_counts": eval_info["prediction_counts"],
        }
        summary_rows.append(row)

        print(
            f"Summary for h={hidden}: "
            f"acc={row['accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"zero_output_rate={row['zero_output_rate']:.2%}"
        )

    out_json = os.path.join(results_dir, f"hidden_size_sweep_{label_mode}.json")
    with open(out_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSaved sweep summary -> {out_json}")
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a quick SNN hidden-size sweep."
    )
    parser.add_argument(
        "--label-mode",
        default=DEFAULT_LABEL_MODE,
        help="Label mode to evaluate."
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_HIDDEN_SIZES,
        help="Hidden sizes to evaluate."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs per hidden-size run."
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=500,
        help="Training samples per epoch for each run."
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=50,
        help="Held-out samples to evaluate per run."
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for checkpoints and outputs."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hidden_size_sweep(
        hidden_sizes=args.hidden_sizes,
        epochs=args.epochs,
        n_train=args.n_train,
        test_samples=args.test_samples,
        label_mode=args.label_mode,
        results_dir=args.results_dir,
    )
