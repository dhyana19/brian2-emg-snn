import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from snn import build_snn

SIM_TIME = 200 * ms
RESULT_DIR = "results"


def run_test(test_name, rates):
    print(f"\nRunning test: {test_name}")

    start_scope()

    net, input_g, hidden, output, syn_out, spike_hidden, spike_out = build_snn()

    # set firing rates for Poisson input neurons
    input_g.rates = rates

    net.run(SIM_TIME)

    # collect spikes
    th = np.array(spike_hidden.t / ms)
    ih = np.array(spike_hidden.i)

    to = np.array(spike_out.t / ms)
    io = np.array(spike_out.i) + int(hidden.N)

    # combine hidden + output spikes
    times = np.concatenate([th, to])
    neurons = np.concatenate([ih, io])

    # plot raster
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(times, neurons, s=3)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index (hidden then output)")
    ax.set_title(f"Raster: {test_name}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f"sanity_{test_name}.png")

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved raster -> {out_path}")
    print("Hidden spikes:", len(th))
    print("Output spikes:", len(to))


def main():

    n_inputs = 8

    # 1️⃣ No input
    run_test(
        "no_input",
        np.zeros(n_inputs) * Hz
    )

    # 2️⃣ All inputs active
    run_test(
        "all_inputs_active",
        np.ones(n_inputs) * 80 * Hz
    )

    # 3️⃣ Single neuron active
    rates = np.zeros(n_inputs)
    rates[0] = 80
    run_test(
        "input_neuron_0",
        rates * Hz
    )

    # 4️⃣ Another neuron active
    rates = np.zeros(n_inputs)
    rates[3] = 80
    run_test(
        "input_neuron_3",
        rates * Hz
    )

    # 5️⃣ Random pattern
    run_test(
        "random_pattern",
        np.random.uniform(10, 80, size=n_inputs) * Hz
    )


if __name__ == "__main__":
    main()