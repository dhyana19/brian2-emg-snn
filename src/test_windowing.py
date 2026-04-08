from windowing import create_windows_from_file
import matplotlib.pyplot as plt

FILE_PATH = (
    "data/EMG_data_for_gestures-master/"
    "01/1_raw_data_13-12_22.03.16.txt"
)

X, y = create_windows_from_file(FILE_PATH)

print("Windows shape:", X.shape)
print("Labels distribution:", {0: (y==0).sum(), 1: (y==1).sum()})

# Plot one random window
i = 10
plt.plot(X[i][:, 0])
plt.title(f"Window {i}, Label = {y[i]}")
plt.show()
