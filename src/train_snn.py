import argparse
import numpy as np
import os
from brian2 import *
from snn import build_snn
from build_dataset import build_full_dataset, split_dataset_by_subject
from experiment_logging import append_experiment_summary
from hidden_temporal_readout import (
    default_prototypes_path,
    extract_hidden_spike_features,
    fit_class_prototypes,
    predict_with_prototypes,
    save_prototypes,
)
from label_modes import DEFAULT_LABEL_MODE, get_class_names, num_classes_for_mode
from sklearn.metrics import accuracy_score, f1_score
from windowing import DEFAULT_SNN_ENCODER_MODE, preprocess_snn_window


# More training per epoch
DEFAULT_N_TRAIN = 3000
DEFAULT_EPOCHS = 25
SIM_TIME = 200 * ms

ETA = 2e-3
W_MAX = 1.0
VERBOSE = False
TEST_SIZE = 0.3
SPLIT_RANDOM_STATE = 42
DEFAULT_HIDDEN = 80
DEFAULT_RESULTS_DIR = "results"
DEFAULT_READOUT_MODE = "output_wta"
DEFAULT_TIME_BINS = 5


def default_weights_path(results_dir, label_mode, n_hidden):
    return os.path.join(results_dir, f"syn_out_w_{label_mode}_h{n_hidden}.npy")


def format_class_counts(y, num_classes):
    counts = np.bincount(y.astype(int), minlength=num_classes)
    return {int(i): int(counts[i]) for i in range(num_classes)}


def train_snn(
    label_mode=DEFAULT_LABEL_MODE,
    n_hidden=DEFAULT_HIDDEN,
    n_train=DEFAULT_N_TRAIN,
    epochs=DEFAULT_EPOCHS,
    readout_mode=DEFAULT_READOUT_MODE,
    time_bins=DEFAULT_TIME_BINS,
    encoder_mode=DEFAULT_SNN_ENCODER_MODE,
    results_dir=DEFAULT_RESULTS_DIR,
    weights_path=None,
    prototypes_path=None,
):

    print("Loading dataset...")
    X, y, subjects = build_full_dataset(label_mode=label_mode)
    num_classes = num_classes_for_mode(label_mode)
    class_names = get_class_names(label_mode)

    train_idx, test_idx = split_dataset_by_subject(
        X,
        y,
        subjects,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    train_subjects = subjects[train_idx]
    test_subjects = subjects[test_idx]

    print("Train size:", len(train_idx))
    print("Held-out test size:", len(test_idx))
    print("Train subjects:", np.unique(train_subjects).tolist())
    print("Held-out subjects:", np.unique(test_subjects).tolist())
    print("Label mode:", label_mode)
    print("Class names:", class_names)
    print("Train class counts:", format_class_counts(y_train, num_classes))
    print("Held-out class counts:", format_class_counts(y[test_idx], num_classes))
    print("Readout mode:", readout_mode)

    if readout_mode == "hidden_temporal_prototype":
        sample_count = min(int(n_train), len(X_train))
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(len(X_train), size=sample_count, replace=False))

        X_fit = X_train[sample_idx]
        y_fit = y_train[sample_idx]

        print("Building SNN feature extractor...")
        print("Temporal bins:", time_bins)
        print("Encoder mode:", encoder_mode)
        print("Prototype training samples:", len(X_fit))

        X_feat = extract_hidden_spike_features(
            X_fit,
            n_hidden=n_hidden,
            n_outputs=num_classes,
            time_bins=time_bins,
            encoder_mode=encoder_mode,
        )

        prototypes = fit_class_prototypes(X_feat, y_fit, num_classes)
        y_pred = predict_with_prototypes(X_feat, prototypes)

        train_acc = accuracy_score(y_fit, y_pred)
        train_f1 = f1_score(y_fit, y_pred, average="macro")
        zero_rate = float(np.mean(np.sum(X_feat, axis=1) == 0.0))

        print(f"Prototype train accuracy: {train_acc:.4f}")
        print(f"Prototype train macro F1: {train_f1:.4f}")
        print(f"Prototype zero-feature rate: {zero_rate:.2%}")

        os.makedirs(results_dir, exist_ok=True)
        if prototypes_path is None:
            prototypes_path = default_prototypes_path(
                results_dir,
                label_mode,
                n_hidden,
                time_bins,
                encoder_mode,
            )

        metadata = save_prototypes(
            prototypes_path,
            prototypes,
            label_mode=label_mode,
            class_names=class_names,
            n_hidden=n_hidden,
            time_bins=time_bins,
            encoder_mode=encoder_mode,
            input_gain=5000.0,
            delta_gain=500.0,
        )

        print(f"Saved hidden temporal prototypes to {prototypes_path}.")
        results = {
            "label_mode": label_mode,
            "class_names": class_names,
            "n_hidden": n_hidden,
            "n_train": n_train,
            "epochs": 1,
            "readout_mode": readout_mode,
            "time_bins": time_bins,
            "encoder_mode": encoder_mode,
            "prototypes_path": prototypes_path,
            "prototype_metadata": metadata,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "accuracy": float(train_acc),
            "macro_f1": float(train_f1),
            "zero_hidden_rate": zero_rate,
        }
        summary_path = append_experiment_summary(
            results,
            results_dir=results_dir,
            script="train_snn.py",
            mode=readout_mode,
        )
        print(f"Updated experiment summary -> {summary_path}")
        return results

    # Compute class weights
    classes_all = np.unique(y_train.astype(int))
    max_class = int(classes_all.max())
    class_counts = np.bincount(y_train.astype(int), minlength=max_class + 1).astype(float)
    class_counts[class_counts == 0] = 1.0
    total_samples = float(len(y_train))
    class_weight = total_samples / (2.0 * class_counts)

    print("Building SNN...")
    net, input_g, hidden, output, syn_out, spike_hidden, spike_out = build_snn(
        n_hidden=n_hidden,
        n_outputs=num_classes,
        encoder_mode=encoder_mode,
    )

    n_hidden = int(hidden.N)
    n_output = int(output.N)

    # -------------------------------------------------
    # OUTPUT WEIGHT INITIALIZATION (stable)
    # -------------------------------------------------
    try:
        rng = np.random.default_rng(42)
        w_init = np.zeros((n_hidden, n_output))
        for k in range(n_output):
            w_init[:, k] = rng.uniform(
                0.05 + k * 0.15,
                0.25 + k * 0.15,
                size=n_hidden,
            )

        syn_out.w[:] = w_init.flatten()

        print("Initialized syn_out.w with random weights.")

    except Exception as e:
        print("Weight initialization warning:", e)

    prev_epoch_avg_output_spikes = np.ones(n_output)
    best_train_acc = 0.0
    best_weights = np.array(syn_out.w[:])

    for epoch in range(epochs):

        correct = 0

        classes = np.unique(y_train)
        max_class = int(classes.max())

        spikes_per_class = np.zeros(max_class + 1)
        counts_per_class = np.zeros(max_class + 1)
        output_spikes_per_class = np.zeros((max_class + 1, n_output))
        zero_output_per_class = np.zeros(max_class + 1, dtype=int)
        zero_output_samples = 0

        prev_hidden_counts = np.array(spike_hidden.count)
        prev_output_counts = np.array(spike_out.count)

        preds_per_class = np.zeros(max_class + 1, dtype=int)

        for step in range(n_train):

            idx = np.random.randint(len(X_train))

            # -------------------------------------------------
            # Prepare EMG window
            # -------------------------------------------------
            window = np.array(X_train[idx], dtype=float)

            if window.ndim == 2 and window.shape[0] == int(input_g.N):
                window = window.T

            T = int(window.shape[0])
            dt = SIM_TIME / float(T)

            scaled_window = preprocess_snn_window(
                window,
                encoder_mode=encoder_mode,
            )

            input_signal = TimedArray(scaled_window.T, dt=dt, name='input_signal')

            # reset neurons
            input_g.v = 0
            hidden.v = 0
            output.v = 0

            net.run(SIM_TIME, namespace={'input_signal': input_signal})

            # -------------------------------------------------
            # Compute spike counts
            # -------------------------------------------------
            current_hidden_counts = np.array(spike_hidden.count)
            hidden_spike_counts = current_hidden_counts - prev_hidden_counts
            prev_hidden_counts = current_hidden_counts

            current_output_counts = np.array(spike_out.count)
            spike_counts = current_output_counts - prev_output_counts
            prev_output_counts = current_output_counts

            total_spikes = float(np.sum(spike_counts))
            norm_spikes = spike_counts.astype(float) / (total_spikes + 1e-6)

            y_pred = int(np.argmax(norm_spikes))
            y_true = int(y_train[idx])

            if y_pred == y_true:
                correct += 1

            # -------------------------------------------------
            # Learning update
            # -------------------------------------------------
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

            spikes_per_class[y_true] += spike_counts.sum()
            output_spikes_per_class[y_true] += spike_counts
            counts_per_class[y_true] += 1

            if float(np.sum(spike_counts)) == 0.0:
                zero_output_samples += 1
                zero_output_per_class[y_true] += 1

            preds_per_class[y_pred] += 1

        # -------------------------------------------------
        # Epoch summary
        # -------------------------------------------------

        train_acc = correct / float(n_train)

        print(f"Epoch {epoch+1} training accuracy: {train_acc:.4f}")

        print(f"Epoch {epoch+1} average spikes per class:")

        for cls in range(len(spikes_per_class)):

            if counts_per_class[cls] > 0:
                avg = spikes_per_class[cls] / counts_per_class[cls]
            else:
                avg = 0.0

            print(f"  Class {cls}: {avg:.2f}")

        print(f"Epoch {epoch+1} average output spikes per true class:")

        for cls in range(len(spikes_per_class)):

            if counts_per_class[cls] > 0:
                avg_vec = output_spikes_per_class[cls] / counts_per_class[cls]
            else:
                avg_vec = np.zeros(n_output)

            print(f"  Class {cls}: {avg_vec.tolist()}")

        zero_rate = zero_output_samples / float(n_train)
        print(
            f"Epoch {epoch+1} zero-output samples: "
            f"{zero_output_samples}/{n_train} ({zero_rate:.2%})"
        )

        print(f"Epoch {epoch+1} zero-output samples by true class:")

        for cls in range(len(zero_output_per_class)):

            if counts_per_class[cls] > 0:
                cls_zero_rate = zero_output_per_class[cls] / counts_per_class[cls]
            else:
                cls_zero_rate = 0.0

            print(
                f"  Class {cls}: {zero_output_per_class[cls]} "
                f"({cls_zero_rate:.2%})"
            )

        print(f"Epoch {epoch+1} prediction distribution:")

        for cls in range(len(preds_per_class)):
            print(f"  Class {cls}: {preds_per_class[cls]}")

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_weights = np.array(syn_out.w[:])
            print(f"  -> New best epoch: {epoch+1} (acc={best_train_acc:.4f})")

        epoch_avg_spikes = output_spikes_per_class.sum(axis=0) / max(float(n_train), 1.0)
        prev_epoch_avg_output_spikes = np.maximum(epoch_avg_spikes, 0.1)

    print("\nTraining complete.")

    os.makedirs(results_dir, exist_ok=True)
    if weights_path is None:
        weights_path = default_weights_path(results_dir, label_mode, n_hidden)
    np.save(weights_path, best_weights)

    print(f"Saved BEST weights from training (acc={best_train_acc:.4f})")
    print(f"Saved trained weights ({len(syn_out.w)} syn_out values) to {weights_path}.")
    results = {
        "label_mode": label_mode,
        "class_names": class_names,
        "n_hidden": n_hidden,
        "n_train": n_train,
        "epochs": epochs,
        "readout_mode": readout_mode,
        "weights_path": weights_path,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
    }
    summary_path = append_experiment_summary(
        results,
        results_dir=results_dir,
        script="train_snn.py",
        mode=readout_mode,
    )
    print(f"Updated experiment summary -> {summary_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train the SNN classifier.")
    parser.add_argument(
        "--label-mode",
        default=DEFAULT_LABEL_MODE,
        help="Label mode to train on."
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
        "--n-train",
        type=int,
        default=DEFAULT_N_TRAIN,
        help="Training samples per epoch."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs."
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_snn(
        label_mode=args.label_mode,
        n_hidden=args.hidden,
        n_train=args.n_train,
        epochs=args.epochs,
        readout_mode=args.readout_mode,
        time_bins=args.time_bins,
        encoder_mode=args.encoder_mode,
        results_dir=args.results_dir,
        weights_path=args.weights_path,
        prototypes_path=args.prototypes_path,
    )
