import numpy as np
from brian2 import *
from snn import build_snn
from windowing import create_windows_from_file


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=0))


FILE_PATH = (
    "data/EMG_data_for_gestures-master/"
    "01/1_raw_data_13-12_22.03.16.txt"
)

# Load one EMG window
X, y = create_windows_from_file(FILE_PATH)

rates = rms(X[0])
rates = rates / rates.max() * 400  # strong input

# Build network
net, input_g, hidden, output, syn_out, spike_hidden, spike_out = build_snn()

# Reset state (IMPORTANT)
hidden.v = 0
output.v = 0

# Set firing rates
input_g.rates = rates * Hz

# Run simulation
net.run(300 * ms)

print("True label:", y[0])
print("Output spikes:", spike_out.count)
print("Predicted label:", int(np.argmax(spike_out.count)))
