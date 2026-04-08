import numpy as np
np.random.seed(42)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
from brian2 import *
from brian2 import seed as brian2_seed
brian2_seed(42)

from common import (
    RESULTS_DIR,
    TEST_SIZE,
    SPLIT_RANDOM_STATE,
    ensure_results_dir,
    load_subject_split,
    plot_confusion_matrix,
    save_json,
)
from label_modes import get_class_names
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from snn import build_snn
from windowing import preprocess_snn_window


SIM_TIME = 200 * ms
N_HIDDEN = 80
N_TRAIN = 1000
EPOCHS = 20
ENCODER_MODE = "envelope"
ETA = 2e-3
W_MAX = 1.0


def save_training_curve(epoch_accuracies, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = np.arange(1, len(epoch_accuracies) + 1)
    ax.plot(epochs, epoch_accuracies, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_and_save_best_weights(X_train, y_train, weights_path):
    class_names = get_class_names("full_6")
    num_classes = len(class_names)

    net, input_g, hidden, output, syn_out, spike_hidden, spike_out = build_snn(
        n_hidden=N_HIDDEN,
        n_outputs=num_classes,
        encoder_mode=ENCODER_MODE,
    )

    n_hidden = int(hidden.N)
    n_output = int(output.N)

    classes_all = np.unique(y_train.astype(int))
    max_class = int(classes_all.max())
    class_counts = np.bincount(y_train.astype(int), minlength=max_class + 1).astype(float)
    class_counts[class_counts == 0] = 1.0
    total_samples = float(len(y_train))
    class_weight = total_samples / (2.0 * class_counts)

    rng = np.random.default_rng(42)
    w_init = np.zeros((n_hidden, n_output))
    for k in range(n_output):
        w_init[:, k] = rng.uniform(
            0.05 + k * 0.15,
            0.25 + k * 0.15,
            size=n_hidden,
        )
    syn_out.w[:] = w_init.flatten()

    best_train_acc = 0.0
    best_weights = np.array(syn_out.w[:])
    epoch_accuracies = []

    for epoch in range(EPOCHS):
        correct = 0
        prev_hidden_counts = np.array(spike_hidden.count)
        prev_output_counts = np.array(spike_out.count)

        for _ in range(N_TRAIN):
            idx = np.random.randint(len(X_train))
            window = np.array(X_train[idx], dtype=float)

            if window.ndim == 2 and window.shape[0] == int(input_g.N):
                window = window.T

            T = int(window.shape[0])
            dt = SIM_TIME / float(T)
            scaled_window = preprocess_snn_window(window, encoder_mode=ENCODER_MODE)
            input_signal = TimedArray(scaled_window.T, dt=dt, name="input_signal")

            input_g.v = 0
            hidden.v = 0
            output.v = 0

            net.run(SIM_TIME, namespace={"input_signal": input_signal})

            current_hidden_counts = np.array(spike_hidden.count)
            hidden_spike_counts = current_hidden_counts - prev_hidden_counts
            prev_hidden_counts = current_hidden_counts

            current_output_counts = np.array(spike_out.count)
            spike_counts = current_output_counts - prev_output_counts
            prev_output_counts = current_output_counts

            total_spikes = float(np.sum(spike_counts))
            norm_spikes = spike_counts.astype(float) / (total_spikes + 1e-6)
            y_true = int(y_train[idx])
            y_pred = int(np.argmax(norm_spikes))

            if y_pred == y_true:
                correct += 1

            w_mat = np.array(syn_out.w[:]).reshape((n_hidden, n_output))
            pre_spikes = hidden_spike_counts.astype(float)
            eta_eff = ETA * class_weight[y_true]
            if float(np.sum(hidden_spike_counts)) < 5.0:
                eta_eff *= 3.0

            winner = int(np.argmax(spike_counts))

            for o in range(n_output):
                if o == y_true:
                    delta = eta_eff * pre_spikes * (1.0 - norm_spikes[o])
                elif o == winner and o != y_true:
                    delta = -eta_eff * 2.0 * pre_spikes * norm_spikes[o]
                else:
                    delta = -eta_eff * 0.5 * pre_spikes * norm_spikes[o]
                w_mat[:, o] += delta

            w_flat = np.clip(w_mat.flatten(), 0.0, W_MAX)
            w_flat += np.random.default_rng().uniform(-0.002, 0.002, size=w_flat.shape)
            w_flat = np.clip(w_flat, 0.0, W_MAX)
            syn_out.w[:] = w_flat

        train_acc = correct / float(N_TRAIN)
        epoch_accuracies.append(float(train_acc))
        print(f"Epoch {epoch + 1} training accuracy: {train_acc:.4f}")
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_weights = np.array(syn_out.w[:])
            print(f"  -> New best epoch: {epoch + 1} (acc={best_train_acc:.4f})")

    np.save(weights_path, best_weights)
    return best_train_acc, epoch_accuracies


def evaluate_saved_weights(X_test, y_test, weights_path):
    class_names = get_class_names("full_6")
    num_classes = len(class_names)

    net, input_g, hidden, output, syn_out, spike_hidden, spike_out = build_snn(
        n_hidden=N_HIDDEN,
        n_outputs=num_classes,
        encoder_mode=ENCODER_MODE,
    )
    syn_out.w[:] = np.load(weights_path).flatten()

    y_true_all = []
    y_pred_all = []
    spike_per_class = [[] for _ in range(num_classes)]
    zero_output_samples = 0

    for i in range(len(y_test)):
        input_g.v = 0
        hidden.v = 0
        output.v = 0

        window = np.array(X_test[i], dtype=float)
        if window.ndim == 2 and window.shape[0] == int(input_g.N):
            window = window.T

        T = int(window.shape[0])
        dt = SIM_TIME / float(T)
        scaled_window = preprocess_snn_window(window, encoder_mode=ENCODER_MODE)
        input_signal = TimedArray(scaled_window.T, dt=dt, name="input_signal")

        prev_output_counts = np.array(spike_out.count)
        net.run(SIM_TIME, namespace={"input_signal": input_signal})
        current_output_counts = np.array(spike_out.count)
        spike_counts = current_output_counts - prev_output_counts

        y_true = int(y_test[i])
        spike_per_class[y_true].append(spike_counts)

        total = float(np.sum(spike_counts))
        if total == 0.0:
            zero_output_samples += 1
        normalized = spike_counts.astype(float) / total if total > 0 else spike_counts.astype(float)
        y_pred = int(np.argmax(normalized))

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

    average_spikes_per_class = []
    for cls in range(num_classes):
        arrs = spike_per_class[cls]
        if arrs:
            mean_vals = np.mean(np.stack(arrs, axis=0), axis=0)
            average_spikes_per_class.append([float(x) for x in mean_vals])
        else:
            average_spikes_per_class.append([0.0 for _ in range(num_classes)])

    return {
        "class_names": class_names,
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")),
        "confusion_matrix": cm,
        "zero_output_rate": float(zero_output_samples / max(len(y_true_all), 1)),
        "average_spikes_per_class": average_spikes_per_class,
    }


def main():
    print("=== Experiment 6: Pure Brian2 SNN (6-class) ===")
    ensure_results_dir()
    split = load_subject_split("full_6")

    weights_path = os.path.join(RESULTS_DIR, "exp6_snn_6class_weights.npy")
    json_path = os.path.join(RESULTS_DIR, "exp6_snn_6class.json")
    cm_path = os.path.join(RESULTS_DIR, "exp6_snn_6class_cm.png")
    curve_path = os.path.join(RESULTS_DIR, "exp6_snn_6class_training_curve.png")

    best_train_acc, epoch_accuracies = train_and_save_best_weights(
        split["X_train"],
        split["y_train"],
        weights_path,
    )
    metrics = evaluate_saved_weights(split["X_test"], split["y_test"], weights_path)
    save_training_curve(
        epoch_accuracies,
        curve_path,
        "Experiment 6 Training Curve",
    )

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["class_names"],
        cm_path,
        "Experiment 6 Confusion Matrix",
    )
    save_json(
        json_path,
        {
            "experiment": "exp6_snn_6class",
            "label_mode": "full_6",
            "readout_mode": "output_wta",
            "n_hidden": N_HIDDEN,
            "epochs": EPOCHS,
            "n_train": N_TRAIN,
            "encoder_mode": ENCODER_MODE,
            "test_size": TEST_SIZE,
            "split_random_state": SPLIT_RANDOM_STATE,
            "train_samples": len(split["y_train"]),
            "test_samples": len(split["y_test"]),
            "best_train_accuracy": best_train_acc,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "zero_output_rate": metrics["zero_output_rate"],
            "average_spikes_per_class": metrics["average_spikes_per_class"],
            "weights_path": weights_path,
            "epoch_accuracies": epoch_accuracies,
            "training_curve_png": curve_path,
            "confusion_matrix_png": cm_path,
        },
    )

    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    print(f"Final macro F1: {metrics['macro_f1']:.4f}")
    print("Saved results to results/exp6_snn_6class.json")


if __name__ == "__main__":
    main()
