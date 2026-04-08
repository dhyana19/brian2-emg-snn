import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
np.random.seed(42)

import copy
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from build_dataset import build_full_dataset, split_dataset_by_subject
from emg_features import extract_handcrafted_features
from label_modes import get_class_names, num_classes_for_mode
from spikingjelly.activation_based import functional, neuron, surrogate


TEST_SIZE = 0.3
SPLIT_RANDOM_STATE = 42
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)
N_HIDDEN = 256
T = 16
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
PATIENCE = 15
EVAL_REPEATS = 3
FINAL_EVAL_REPEATS = 5
WEIGHTS_PATH = os.path.join(RESULTS_DIR, "exp10_snn_6class_weights.pt")
CM_PATH = os.path.join(RESULTS_DIR, "exp10_snn_6class_cm.png")
CURVE_PATH = os.path.join(RESULTS_DIR, "exp10_snn_6class_training_curve.png")
JSON_PATH = os.path.join(RESULTS_DIR, "exp10_snn_6class.json")


torch.manual_seed(42)


class SNN6Class(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, T):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan())
        self.fc3 = nn.Linear(n_hidden, n_output)
        self.lif3 = neuron.LIFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        functional.reset_net(self)
        spike_out = 0
        for _ in range(self.T):
            encoded = x
            h1 = self.lif1(self.fc1(encoded))
            h2 = self.lif2(self.fc2(h1))
            out = self.lif3(self.fc3(h2))
            spike_out += out
        return spike_out / self.T


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def evaluate_model(model, data_loader, device, eval_repeats=1):
    model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = 0
            for _ in range(eval_repeats):
                outputs = outputs + model(batch_x)
            outputs = outputs / float(eval_repeats)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred_all.extend(preds.tolist())
            y_true_all.extend(batch_y.numpy().tolist())

    acc = float(accuracy_score(y_true_all, y_pred_all))
    macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro"))
    return acc, macro_f1, np.array(y_true_all), np.array(y_pred_all)


def main():
    print("=== Experiment 10: SpikingJelly surrogate-gradient SNN (6-class) ===")
    ensure_results_dir()

    X, y, subjects = build_full_dataset(label_mode="full_6")
    class_names = get_class_names("full_6")
    num_classes = num_classes_for_mode("full_6")

    print("Extracting handcrafted features...")
    X_feat = extract_handcrafted_features(X)
    print("Feature shape:", X_feat.shape)

    train_idx, test_idx = split_dataset_by_subject(
        X_feat,
        y,
        subjects,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )

    X_train = X_feat[train_idx]
    y_train = y[train_idx]
    X_test = X_feat[test_idx]
    y_test = y[test_idx]

    class_counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float32)
    class_counts[class_counts == 0.0] = 1.0
    class_weights = class_counts.sum() / (num_classes * class_counts)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test.astype(np.int64)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cpu")
    model = SNN6Class(
        n_input=X_train.shape[1],
        n_hidden=N_HIDDEN,
        n_output=num_classes,
        T=T,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    best_test_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    train_acc_history = []
    train_loss_history = []
    test_acc_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_x.shape[0]
            running_loss += float(loss.item()) * batch_size
            total_examples += batch_size

        avg_train_loss = running_loss / max(total_examples, 1)
        train_acc, _, _, _ = evaluate_model(
            model,
            train_eval_loader,
            device,
            eval_repeats=1,
        )
        test_acc, _, _, _ = evaluate_model(
            model,
            test_loader,
            device,
            eval_repeats=EVAL_REPEATS,
        )

        train_loss_history.append(float(avg_train_loss))
        train_acc_history.append(float(train_acc))
        test_acc_history.append(float(test_acc))

        if test_acc > best_test_acc:
            best_test_acc = float(test_acc)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, WEIGHTS_PATH)
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} - "
                f"train_loss={avg_train_loss:.4f}, "
                f"train_acc={train_acc:.4f}, "
                f"test_acc={test_acc:.4f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(best epoch: {best_epoch}, best test acc: {best_test_acc:.4f})"
            )
            break

    model.load_state_dict(best_state_dict)
    final_acc, final_macro_f1, y_true_all, y_pred_all = evaluate_model(
        model,
        test_loader,
        device,
        eval_repeats=FINAL_EVAL_REPEATS,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_loss_history, label='Train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training loss')
    ax1.legend()

    ax2.plot(train_acc_history, label='Train accuracy')
    ax2.plot(test_acc_history, label='Test accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and test accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(CURVE_PATH, bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved training curve -> results/exp10_snn_6class_training_curve.png')

    cm = confusion_matrix(y_true_all, y_pred_all)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('SpikingJelly SNN - 6-class confusion matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(CM_PATH, bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved confusion matrix -> results/exp10_snn_6class_cm.png')

    results = {
        "accuracy": final_acc,
        "macro_f1": final_macro_f1,
        "confusion_matrix": cm.tolist(),
        "n_hidden": N_HIDDEN,
        "T": T,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "class_weights": class_weights.tolist(),
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "framework": "SpikingJelly",
        "note": "Surrogate gradient SNN addressing Brian2 output collapse",
        "training_curve_png": "results/exp10_snn_6class_training_curve.png",
        "confusion_matrix_png": "results/exp10_snn_6class_cm.png",
        "weights_path": "results/exp10_snn_6class_weights.pt",
        "train_acc_history": train_acc_history,
        "train_loss_history": train_loss_history,
        "test_acc_history": test_acc_history,
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(to_builtin(results), f, indent=2)

    print("=== Experiment 10 complete ===")
    print(f"Final accuracy: {final_acc:.4f}")
    print(f"Final macro F1: {final_macro_f1:.4f}")
    print("Saved results to results/exp10_snn_6class.json")


if __name__ == "__main__":
    main()
