import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = (
    "data/EMG_data_for_gestures-master/"
    "01/1_raw_data_13-12_22.03.16.txt"
)

data = np.loadtxt(FILE_PATH, delimiter=None, skiprows=1)

print("Data shape:", data.shape)

time = data[:, 0]
emg = data[:, 1:9]      # 8 EMG channels
labels = data[:, 9]

print("Unique labels:", np.unique(labels))
print("EMG min / max:", emg.min(), emg.max())

plt.figure(figsize=(10, 3))
plt.plot(emg[:2000, 0])
plt.title("EMG Channel 0 (first 2000 samples)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
