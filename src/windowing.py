import numpy as np
from label_modes import DEFAULT_LABEL_MODE, get_label_map


DEFAULT_SMOOTHING_KERNEL = 5
DEFAULT_SNN_INPUT_GAIN = 5000.0
DEFAULT_SNN_DELTA_GAIN = 500.0
DEFAULT_SNN_ENCODER_MODE = "envelope"


def map_label(label, label_mode=DEFAULT_LABEL_MODE):
    """
    Map raw EMG labels to experiment-specific class ids.
    """
    label_map = get_label_map(label_mode)
    return label_map.get(int(label))


def create_windows_from_file(
    file_path,
    window_size=200,
    step_size=100,
    label_mode=DEFAULT_LABEL_MODE
):
    """
    Load one EMG file and convert it into windows.

    Returns:
        X_windows: (N, window_size, 8)
        y_windows: (N,)
    """

    # Robust loading: tolerate malformed rows
    data = np.genfromtxt(
        file_path,
        skip_header=1,
        usecols=range(10),
        invalid_raise=False
    )

    # Drop rows with missing values
    data = data[~np.isnan(data).any(axis=1)]

    # Columns
    emg = data[:, 1:9]     # 8 EMG channels
    labels = data[:, 9]   # raw labels

    X_windows = []
    y_windows = []

    for start in range(0, len(emg) - window_size + 1, step_size):

        end = start + window_size

        window_emg = emg[start:end].astype(float)

        window_labels = labels[start:end].astype(int)

        raw_label = np.bincount(window_labels).argmax()
        mapped_label = map_label(raw_label, label_mode=label_mode)

        if mapped_label is None:
            continue

        X_windows.append(window_emg)
        y_windows.append(mapped_label)

    return np.array(X_windows), np.array(y_windows)


def preprocess_snn_window(
    window,
    smoothing_kernel=DEFAULT_SMOOTHING_KERNEL,
    input_gain=DEFAULT_SNN_INPUT_GAIN,
    delta_gain=DEFAULT_SNN_DELTA_GAIN,
    encoder_mode=DEFAULT_SNN_ENCODER_MODE,
):
    """
    Convert a raw EMG window into a positive drive signal for the SNN.
    """

    window = np.asarray(window, dtype=float)

    if window.ndim != 2:
        raise ValueError(f"Expected a 2D window, got shape {window.shape}.")

    window_rect = np.abs(window)
    window_delta = np.abs(
        np.diff(window, axis=0, prepend=window[:1, :])
    )

    if smoothing_kernel > 1:
        kernel = np.ones(int(smoothing_kernel), dtype=float) / float(smoothing_kernel)
        envelope = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"),
            axis=0,
            arr=window_rect
        )
        delta = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"),
            axis=0,
            arr=window_delta
        )
    else:
        envelope = window_rect
        delta = window_delta

    if encoder_mode == "envelope":
        return envelope * float(input_gain)

    if encoder_mode == "envelope_delta":
        envelope_drive = envelope * float(input_gain)
        delta_drive = delta * float(delta_gain)
        return np.concatenate([envelope_drive, delta_drive], axis=1)

    raise ValueError(
        f"Unknown encoder_mode '{encoder_mode}'. "
        "Valid options: envelope, envelope_delta."
    )


def get_snn_input_channels(
    n_emg_channels=8,
    encoder_mode=DEFAULT_SNN_ENCODER_MODE
):
    if encoder_mode == "envelope":
        return int(n_emg_channels)
    if encoder_mode == "envelope_delta":
        return int(2 * n_emg_channels)
    raise ValueError(
        f"Unknown encoder_mode '{encoder_mode}'. "
        "Valid options: envelope, envelope_delta."
    )
