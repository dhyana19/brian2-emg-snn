import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from windowing import create_windows_from_file
from label_modes import DEFAULT_LABEL_MODE

DATA_ROOT = "data/EMG_data_for_gestures-master"


def build_full_dataset(label_mode=DEFAULT_LABEL_MODE):
    X_all = []
    y_all = []
    subjects = []

    subject_folders = sorted(os.listdir(DATA_ROOT))

    for subj_id, subj_folder in enumerate(subject_folders):
        subj_path = os.path.join(DATA_ROOT, subj_folder)

        if not os.path.isdir(subj_path):
            continue

        for fname in os.listdir(subj_path):
            if not fname.endswith(".txt"):
                continue

            file_path = os.path.join(subj_path, fname)

            try:
                X, y = create_windows_from_file(file_path, label_mode=label_mode)

                if len(y) == 0:
                    continue

                X_all.append(X)
                y_all.append(y)
                subjects.append(np.full(len(y), subj_id))

            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")

    return (
        np.concatenate(X_all),
        np.concatenate(y_all),
        np.concatenate(subjects),
    )


def split_dataset_by_subject(
    X,
    y,
    subjects,
    test_size=0.3,
    random_state=42
):
    splitter = GroupShuffleSplit(
        test_size=test_size,
        n_splits=1,
        random_state=random_state
    )

    train_idx, test_idx = next(splitter.split(X, y, groups=subjects))
    return train_idx, test_idx


if __name__ == "__main__":
    X, y, s = build_full_dataset()

    print("Final dataset shape:", X.shape)
    print("Label distribution:", {
        0: int((y == 0).sum()),
        1: int((y == 1).sum())
    })
    print("Number of subjects:", len(np.unique(s)))
