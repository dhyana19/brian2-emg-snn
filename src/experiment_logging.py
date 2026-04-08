import csv
import os
from datetime import datetime, timezone


SUMMARY_COLUMNS = [
    "timestamp_utc",
    "script",
    "mode",
    "label_mode",
    "class_names",
    "feature_mode",
    "readout_mode",
    "encoder_mode",
    "n_hidden",
    "time_bins",
    "train_samples",
    "test_samples",
    "train_size",
    "test_size",
    "accuracy",
    "macro_f1",
    "zero_hidden_rate",
    "zero_output_rate",
    "model_path",
    "results_json",
    "confusion_matrix_png",
]


def _stringify(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    return str(value)


def append_experiment_summary(results, results_dir, script, mode):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "experiment_summary.csv")

    row = {
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "script": script,
        "mode": mode,
        "label_mode": results.get("label_mode"),
        "class_names": results.get("class_names"),
        "feature_mode": results.get("feature_mode"),
        "readout_mode": results.get("readout_mode"),
        "encoder_mode": results.get("encoder_mode"),
        "n_hidden": results.get("n_hidden"),
        "time_bins": results.get("time_bins"),
        "train_samples": results.get("train_samples"),
        "test_samples": results.get("test_samples") or results.get("evaluated_samples"),
        "train_size": results.get("train_size"),
        "test_size": results.get("test_size"),
        "accuracy": results.get("accuracy"),
        "macro_f1": results.get("macro_f1"),
        "zero_hidden_rate": (
            results.get("test_zero_hidden_rate")
            if results.get("test_zero_hidden_rate") is not None
            else results.get("zero_hidden_rate")
        ),
        "zero_output_rate": results.get("zero_output_rate"),
        "model_path": (
            results.get("model_path")
            or results.get("weights_path")
            or results.get("checkpoint_path")
            or results.get("prototypes_path")
        ),
        "results_json": results.get("results_json"),
        "confusion_matrix_png": results.get("confusion_matrix_png"),
    }

    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: _stringify(row.get(k)) for k in SUMMARY_COLUMNS})

    return summary_path
