import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from snn import build_snn
from build_dataset import build_full_dataset, split_dataset_by_subject
from experiment_logging import append_experiment_summary
from hidden_temporal_readout import (
    default_prototypes_path,
    extract_hidden_spike_features,
    load_prototypes,
    predict_with_prototypes,
)
from label_modes import DEFAULT_LABEL_MODE, get_class_names, num_classes_for_mode
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from windowing import DEFAULT_SNN_ENCODER_MODE, preprocess_snn_window


DEFAULT_TEST_SAMPLES = 50
SIM_TIME = 200 * ms
TEST_SIZE = 0.3
SPLIT_RANDOM_STATE = 42
EVAL_RANDOM_STATE = 42
DEFAULT_HIDDEN = 80
DEFAULT_RESULTS_DIR = "results"
DEFAULT_READOUT_MODE = "output_wta"
DEFAULT_TIME_BINS = 5


def default_weights_path(results_dir, label_mode, n_hidden):
    return os.path.join(results_dir, f"syn_out_w_{label_mode}_h{n_hidden}.npy")


def format_class_counts(y, num_classes):
    counts = np.bincount(y.astype(int), minlength=num_classes)
    return {int(i): int(counts[i]) for i in range(num_classes)}


def estimate_baseline_ops_per_sample(window, num_classes):
    sample = np.array(window, dtype=float)

    if sample.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {sample.shape}")

    n_channels = int(sample.shape[1])
    n_features = n_channels * 4  # RMS, MAV, waveform length, zero crossings
    return float(n_features * num_classes)


def estimate_efficiency(
    total_hidden_spikes,
    total_output_spikes,
    n_outputs,
    evaluated_samples,
    baseline_ops_per_sample
):
    hidden_to_output_events = float(total_hidden_spikes) * float(n_outputs)
    output_lateral_events = float(total_output_spikes) * float(max(n_outputs - 1, 0))
    snn_synaptic_events = hidden_to_output_events + output_lateral_events
    baseline_total_ops = float(baseline_ops_per_sample) * float(evaluated_samples)

    if snn_synaptic_events > 0.0:
        ratio = baseline_total_ops / snn_synaptic_events
    else:
        ratio = None

    return {
        "baseline_ops_per_sample": float(baseline_ops_per_sample),
        "baseline_total_ops": float(baseline_total_ops),
        "snn_total_synaptic_events": float(snn_synaptic_events),
        "baseline_ops_over_snn_events": None if ratio is None else float(ratio),
    }


def load_trained_syn_out_weights(syn_out, weights_path):

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Missing trained weights at {weights_path}. "
            "Run src/train_snn.py first."
        )

    weights = np.load(weights_path).flatten()
    expected = len(syn_out.w)

    if len(weights) != expected:
        raise ValueError(
            f"Checkpoint shape mismatch for syn_out.w: "
            f"expected {expected} values, found {len(weights)} in {weights_path}. "
            "Delete the stale checkpoint and retrain with the current network."
        )

    syn_out.w[:] = weights
    return expected


def eval_snn(
    label_mode=DEFAULT_LABEL_MODE,
    n_hidden=DEFAULT_HIDDEN,
    test_samples=DEFAULT_TEST_SAMPLES,
    readout_mode=DEFAULT_READOUT_MODE,
    time_bins=DEFAULT_TIME_BINS,
    encoder_mode=DEFAULT_SNN_ENCODER_MODE,
    results_dir=DEFAULT_RESULTS_DIR,
    weights_path=None,
    prototypes_path=None,
    artifact_suffix=None
):

    print("Loading dataset...")
    X, y, subjects = build_full_dataset(label_mode=label_mode)
    num_classes = num_classes_for_mode(label_mode)
    class_names = get_class_names(label_mode)

    _, test_idx = split_dataset_by_subject(
        X,
        y,
        subjects,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE
    )

    X_test = X[test_idx]
    y_test = y[test_idx]
    test_subjects = subjects[test_idx]

    print("Held-out test size:", len(y_test))
    print("Held-out subjects:", np.unique(test_subjects).tolist())
    print("Label mode:", label_mode)
    print("Class names:", class_names)
    print("Held-out class counts:", format_class_counts(y_test, num_classes))
    print("Readout mode:", readout_mode)

    if readout_mode == "hidden_temporal_prototype":
        if test_samples is None or test_samples >= len(y_test):
            eval_indices = np.arange(len(y_test))
        else:
            rng = np.random.default_rng(EVAL_RANDOM_STATE)
            eval_indices = np.sort(rng.choice(len(y_test), size=test_samples, replace=False))

        if prototypes_path is None:
            prototypes_path = default_prototypes_path(
                results_dir,
                label_mode,
                n_hidden,
                time_bins,
                encoder_mode,
            )

        prototypes, metadata = load_prototypes(prototypes_path)

        print("Temporal bins:", time_bins)
        print("Encoder mode:", encoder_mode)
        print("Loaded prototypes from:", prototypes_path)
        print("Evaluating samples:", len(eval_indices))

        X_eval = X_test[eval_indices]
        y_eval = y_test[eval_indices]

        X_feat = extract_hidden_spike_features(
            X_eval,
            n_hidden=n_hidden,
            n_outputs=num_classes,
            time_bins=time_bins,
            encoder_mode=encoder_mode,
        )
        y_pred = predict_with_prototypes(X_feat, prototypes)

        labels = np.arange(num_classes)
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average="macro")
        cm = confusion_matrix(y_eval, y_pred, labels=labels)
        clf_report = classification_report(
            y_eval,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )

        os.makedirs(results_dir, exist_ok=True)
        if artifact_suffix is None:
            artifact_suffix = (
                f"_{label_mode}_{readout_mode}_{encoder_mode}_"
                f"h{n_hidden}_tb{time_bins}"
            )

        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax)

        out_path = os.path.join(results_dir, f"confusion_matrix{artifact_suffix}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        pred_counts = np.bincount(y_pred, minlength=num_classes)
        zero_hidden_rate = float(np.mean(np.sum(X_feat, axis=1) == 0.0))

        results = {
            "label_mode": label_mode,
            "class_names": class_names,
            "n_hidden": int(n_hidden),
            "readout_mode": readout_mode,
            "time_bins": int(time_bins),
            "encoder_mode": encoder_mode,
            "prototype_metadata": metadata,
            "evaluated_samples": int(len(eval_indices)),
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_report,
            "prediction_counts": pred_counts.tolist(),
            "zero_hidden_rate": zero_hidden_rate,
        }

        out_json = os.path.join(results_dir, f"results{artifact_suffix}.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

        results["results_json"] = out_json
        results["confusion_matrix_png"] = out_path
        summary_path = append_experiment_summary(
            results,
            results_dir=results_dir,
            script="eval_snn.py",
            mode=readout_mode,
        )

        print("\n===== SNN TEST RESULTS =====")
        print(f"Hidden units : {n_hidden}")
        print(f"Accuracy     : {acc:.4f}")
        print(f"Macro F1     : {f1:.4f}")
        print(f"Zero hidden feature rate: {zero_hidden_rate:.2%}")
        print("Confusion Matrix:")
        print(cm)
        print(f"\nSaved confusion matrix -> {out_path}")
        print(f"Saved evaluation results -> {out_json}")
        print(f"Updated experiment summary -> {summary_path}")
        return results

    print("Building SNN...")
    net, input_g, hidden, output, syn_out, spike_hidden, spike_out = build_snn(
        n_hidden=n_hidden,
        n_outputs=num_classes,
        encoder_mode=encoder_mode,
    )

    n_hidden = int(hidden.N)
    num_classes = int(output.N)

    if weights_path is None:
        weights_path = default_weights_path(results_dir, label_mode, n_hidden)

    expected_weights = load_trained_syn_out_weights(syn_out, weights_path)
    print(f"Loaded trained weights for evaluation ({expected_weights} syn_out values).")

    if test_samples is None or test_samples >= len(y_test):
        eval_indices = np.arange(len(y_test))
    else:
        rng = np.random.default_rng(EVAL_RANDOM_STATE)
        eval_indices = np.sort(rng.choice(len(y_test), size=test_samples, replace=False))

    print("Evaluating samples:", len(eval_indices))

    y_true_all = []
    y_pred_all = []
    spike_per_class = [[] for _ in range(num_classes)]
    zero_output_samples = 0
    zero_output_per_class = np.zeros(num_classes, dtype=int)
    total_hidden_spikes = 0.0

    for i in eval_indices:

        input_g.v = 0
        hidden.v = 0
        output.v = 0

        window = np.array(X_test[i], dtype=float)

        if window.ndim == 2 and window.shape[0] == int(input_g.N):
            window = window.T

        T = int(window.shape[0])
        dt = SIM_TIME / float(T)

        scaled_window = preprocess_snn_window(
            window,
            encoder_mode=encoder_mode,
        )
        input_signal = TimedArray(scaled_window.T, dt=dt, name="input_signal")

        prev_hidden_counts = np.array(spike_hidden.count)
        prev_output_counts = np.array(spike_out.count)
        net.run(SIM_TIME, namespace={"input_signal": input_signal})

        current_hidden_counts = np.array(spike_hidden.count)
        hidden_spike_counts = current_hidden_counts - prev_hidden_counts
        total_hidden_spikes += float(np.sum(hidden_spike_counts))

        current_output_counts = np.array(spike_out.count)
        spike_counts = current_output_counts - prev_output_counts

        y_true = int(y_test[i])
        spike_per_class[y_true].append(spike_counts)

        total = float(np.sum(spike_counts))
        if total == 0.0:
            zero_output_samples += 1
            zero_output_per_class[y_true] += 1

        if total > 0:
            normalized = spike_counts.astype(float) / total
        else:
            normalized = spike_counts.astype(float)

        y_pred = int(np.argmax(normalized))

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    labels = np.arange(num_classes)
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

    clf_report = classification_report(
        y_true_all,
        y_pred_all,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    os.makedirs(results_dir, exist_ok=True)
    if artifact_suffix is None:
        artifact_suffix = f"_{label_mode}_h{n_hidden}"

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(labels)
    ax.set_yticks(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)

    out_path = os.path.join(results_dir, f"confusion_matrix{artifact_suffix}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    pred_counts = np.bincount(y_pred_all, minlength=num_classes)
    flat_spike_counts = [
        np.array(spike_counts, dtype=float)
        for cls_spikes in spike_per_class
        for spike_counts in cls_spikes
    ]
    if flat_spike_counts:
        total_spikes_per_output = np.sum(np.stack(flat_spike_counts), axis=0)
    else:
        total_spikes_per_output = np.zeros(num_classes)

    fig, ax = plt.subplots()
    ax.bar(labels, pred_counts)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Count")
    plt.title("Prediction Distribution")

    out_hist = os.path.join(results_dir, f"prediction_distribution{artifact_suffix}.png")
    plt.savefig(out_hist, bbox_inches="tight")
    plt.close(fig)

    average_spikes_per_class = []
    for c in range(num_classes):
        arrs = spike_per_class[c]
        if len(arrs) > 0:
            mean_vals = np.mean(np.stack(arrs, axis=0), axis=0)
            average_spikes_per_class.append([float(x) for x in mean_vals])
        else:
            average_spikes_per_class.append([0.0 for _ in range(num_classes)])

    baseline_ops_per_sample = estimate_baseline_ops_per_sample(X_test[eval_indices[0]], num_classes)
    efficiency_estimate = estimate_efficiency(
        total_hidden_spikes=total_hidden_spikes,
        total_output_spikes=float(np.sum(total_spikes_per_output)),
        n_outputs=num_classes,
        evaluated_samples=len(eval_indices),
        baseline_ops_per_sample=baseline_ops_per_sample,
    )

    results = {
        "label_mode": label_mode,
        "class_names": class_names,
        "n_hidden": int(n_hidden),
        "checkpoint_path": weights_path,
        "evaluated_samples": int(len(eval_indices)),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        "prediction_counts": pred_counts.tolist(),
        "average_spikes_per_class": average_spikes_per_class,
        "total_spikes_per_output": [float(x) for x in total_spikes_per_output],
        "zero_output_samples": int(zero_output_samples),
        "zero_output_rate": float(zero_output_samples / max(len(y_true_all), 1)),
        "zero_output_samples_by_true_class": zero_output_per_class.tolist(),
        "efficiency_estimate": efficiency_estimate,
    }

    out_json = os.path.join(results_dir, f"results{artifact_suffix}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    results["results_json"] = out_json
    results["confusion_matrix_png"] = out_path
    summary_path = append_experiment_summary(
        results,
        results_dir=results_dir,
        script="eval_snn.py",
        mode=readout_mode,
    )

    print("\n===== SNN TEST RESULTS =====")
    print(f"Hidden units : {n_hidden}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Prediction distribution:", pred_counts)
    print("Average output spikes per class:", average_spikes_per_class)
    print("Total spikes per output neuron:", [float(x) for x in total_spikes_per_output])
    print(
        "Zero-output samples:",
        f"{zero_output_samples}/{len(y_true_all)} "
        f"({zero_output_samples / max(len(y_true_all), 1):.2%})"
    )
    print("Zero-output samples by true class:", zero_output_per_class.tolist())
    print("Efficiency estimate:", efficiency_estimate)
    print(f"\nSaved confusion matrix -> {out_path}")
    print(f"Saved evaluation results -> {out_json}")
    print(f"Updated experiment summary -> {summary_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the SNN classifier.")
    parser.add_argument(
        "--label-mode",
        default=DEFAULT_LABEL_MODE,
        help="Label mode to evaluate."
    )
    parser.add_argument(
        "--readout-mode",
        default=DEFAULT_READOUT_MODE,
        help="Readout mode: output_wta or hidden_temporal_prototype."
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=DEFAULT_TIME_BINS,
        help="Temporal bins per hidden neuron for hidden_temporal_prototype mode."
    )
    parser.add_argument(
        "--encoder-mode",
        default=DEFAULT_SNN_ENCODER_MODE,
        help="Input encoder mode for hidden_temporal_prototype mode."
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=DEFAULT_HIDDEN,
        help="Number of hidden neurons."
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_TEST_SAMPLES,
        help="Number of held-out samples to evaluate."
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for checkpoints and outputs."
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Optional explicit checkpoint path."
    )
    parser.add_argument(
        "--prototypes-path",
        default=None,
        help="Optional explicit prototype path for hidden_temporal_prototype mode."
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Optional suffix for saved plots and JSON outputs."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_snn(
        label_mode=args.label_mode,
        n_hidden=args.hidden,
        test_samples=args.test_samples,
        readout_mode=args.readout_mode,
        time_bins=args.time_bins,
        encoder_mode=args.encoder_mode,
        results_dir=args.results_dir,
        weights_path=args.weights_path,
        prototypes_path=args.prototypes_path,
        artifact_suffix=args.artifact_suffix,
    )
