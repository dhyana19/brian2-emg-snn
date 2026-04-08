import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from snn import build_snn
from build_dataset import build_full_dataset

# Config (match eval_snn)
TEST_START = 1000
TEST_SAMPLES = 50  # run 50-sample diagnostics
SIM_TIME = 200 * ms
R_MAX = 300


def encode_temporal_rates(window, R_max=R_MAX):
    w = np.array(window, dtype=float)
    mins = w.min(axis=0)
    maxs = w.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    norm = (w - mins[np.newaxis, :]) / ranges[np.newaxis, :]
    return norm * float(R_max)


if __name__ == "__main__":
    print("Loading dataset...")
    X, y, _ = build_full_dataset()

    print("Building SNN...")
    net, input_g, hidden, output, syn_out, spike_hidden, spike_out = build_snn()

    # Load weights if present
    weights_path = os.path.join("results", "syn_out_w.npy")
    if os.path.exists(weights_path):
        syn_out.w[:] = np.load(weights_path)
        print("Loaded weights from results/syn_out_w.npy")

    num_classes = int(np.max(y)) + 1
    preds = []
    trues = []

    # accumulators for per-class output spike counts (sum over samples)
    per_class_output_sum = {c: np.zeros(int(output.N), dtype=float) for c in range(num_classes)}
    per_class_counts = {c: 0 for c in range(num_classes)}

    first_correct_saved = False
    first_incorrect_saved = False

    os.makedirs("results", exist_ok=True)

    for idx_i, i in enumerate(range(TEST_START, TEST_START + TEST_SAMPLES)):
        hidden.v = 0
        output.v = 0

        # use raw EMG as a TimedArray-driven input and run once for the window
        window = np.array(X[i], dtype=float)
        if window.ndim == 2 and window.shape[0] == int(input_g.N) and window.shape[1] != int(input_g.N):
            window = window.T
        T = int(window.shape[0]) if window.ndim >= 2 else 1
        dt = SIM_TIME / float(T)

        prev_out = np.array(spike_out.count)

        # compute numeric start time (seconds) as last recorded monitor time (fallback 0.0)
        def _last_time_seconds(mon):
            try:
                if len(mon.t) > 0:
                    return float(np.max(np.array(mon.t / second)))
            except Exception:
                pass
            return 0.0

        t0 = max(_last_time_seconds(spike_hidden), _last_time_seconds(spike_out))

        # Print TimedArray stats (raw and after gain applied in `snn.py`)
        raw = window.T.astype(float)
        def rms(a):
            return float(np.sqrt(np.mean(a ** 2)))

        raw_min = float(np.min(raw))
        raw_max = float(np.max(raw))
        raw_mean = float(np.mean(raw))
        raw_rms = rms(raw)

        gain = 50.0
        scaled = gain * raw
        scaled_min = float(np.min(scaled))
        scaled_max = float(np.max(scaled))
        scaled_mean = float(np.mean(scaled))
        scaled_rms = rms(scaled)

        print('\nTimedArray (raw) stats for sample', i)
        print(f'  min={raw_min:.6g}, max={raw_max:.6g}, mean={raw_mean:.6g}, rms={raw_rms:.6g}')
        print('TimedArray (after gain x50) numeric stats')
        print(f'  min={scaled_min:.6g}, max={scaled_max:.6g}, mean={scaled_mean:.6g}, rms={scaled_rms:.6g}')

        input_signal = TimedArray(window.T, dt=dt)
        net.run(SIM_TIME, namespace={"input_signal": input_signal})

        # compute numeric end time (seconds) from monitors
        t1 = max(_last_time_seconds(spike_hidden), _last_time_seconds(spike_out))

        cur_out = np.array(spike_out.count)
        spike_counts = cur_out - prev_out

        y_true = int(y[i])
        y_pred = int(np.argmax(spike_counts.astype(float) / (np.sum(spike_counts) + 1e-9)))

        preds.append(y_pred)
        trues.append(y_true)

        per_class_output_sum[y_true] += spike_counts
        per_class_counts[y_true] += 1

        # collect spike times within this interval for raster/timeline
        th = np.array(spike_hidden.t / second)
        ih = np.array(spike_hidden.i)
        mask_h = (th >= t0) & (th <= t1)
        th_rel = (th[mask_h] - t0) * 1000.0
        ih_rel = ih[mask_h]

        to = np.array(spike_out.t / second)
        io = np.array(spike_out.i)
        mask_o = (to >= t0) & (to <= t1)
        to_rel = (to[mask_o] - t0) * 1000.0
        io_rel = io[mask_o]

        # Save raster + timeline for first correct / incorrect
        if (not first_correct_saved) and (y_pred == y_true):
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                offset_io = io_rel + int(hidden.N)
                times = np.concatenate([th_rel, to_rel])
                neurons = np.concatenate([ih_rel, offset_io])
                ax.scatter(times, neurons, s=2)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Neuron index (hidden then output)')
                ax.set_title(f'Spike raster CORRECT sample {i}')
                out_raster = os.path.join('results', f'spike_raster_correct_{i}.png')
                fig.tight_layout()
                fig.savefig(out_raster, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved raster for correct sample {i} -> {out_raster}')

                # output timeline (per-timestep counts)
                bins = np.linspace(0.0, float(SIM_TIME / ms), T + 1)
                counts_per_timestep = np.zeros((T, int(output.N)), dtype=int)
                all_to = to[mask_o]
                all_io = io[mask_o]
                for o in range(int(output.N)):
                    times_o = (all_to[all_io == o] - t0) * 1000.0
                    hist, _ = np.histogram(times_o, bins=bins)
                    counts_per_timestep[:, o] = hist
                fig, ax = plt.subplots()
                ax.plot(np.arange(T), counts_per_timestep.sum(axis=1), marker='o')
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Output spikes (sum)')
                ax.set_title(f'Output spike timeline CORRECT sample {i}')
                out_timeline = os.path.join('results', f'output_timeline_correct_{i}.png')
                fig.tight_layout()
                fig.savefig(out_timeline, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved output timeline for correct sample {i} -> {out_timeline}')
            except Exception as e:
                print('Warning saving correct sample visualizations:', e)
            first_correct_saved = True

        if (not first_incorrect_saved) and (y_pred != y_true):
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                offset_io = io_rel + int(hidden.N)
                times = np.concatenate([th_rel, to_rel])
                neurons = np.concatenate([ih_rel, offset_io])
                ax.scatter(times, neurons, s=2)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Neuron index (hidden then output)')
                ax.set_title(f'Spike raster INCORRECT sample {i}')
                out_raster = os.path.join('results', f'spike_raster_incorrect_{i}.png')
                fig.tight_layout()
                fig.savefig(out_raster, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved raster for incorrect sample {i} -> {out_raster}')

                bins = np.linspace(0.0, float(SIM_TIME / ms), T + 1)
                counts_per_timestep = np.zeros((T, int(output.N)), dtype=int)
                all_to = to[mask_o]
                all_io = io[mask_o]
                for o in range(int(output.N)):
                    times_o = (all_to[all_io == o] - t0) * 1000.0
                    hist, _ = np.histogram(times_o, bins=bins)
                    counts_per_timestep[:, o] = hist
                fig, ax = plt.subplots()
                ax.plot(np.arange(T), counts_per_timestep.sum(axis=1), marker='o')
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Output spikes (sum)')
                ax.set_title(f'Output spike timeline INCORRECT sample {i}')
                out_timeline = os.path.join('results', f'output_timeline_incorrect_{i}.png')
                fig.tight_layout()
                fig.savefig(out_timeline, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved output timeline for incorrect sample {i} -> {out_timeline}')
            except Exception as e:
                print('Warning saving incorrect sample visualizations:', e)
            first_incorrect_saved = True

        # early exit if both saved
        if first_correct_saved and first_incorrect_saved:
            pass

    preds = np.array(preds)
    trues = np.array(trues)

    # Prediction distribution per class
    pred_counts = np.bincount(preds, minlength=num_classes)
    print('\nPrediction distribution (per class):')
    for c in range(num_classes):
        print(f'  Class {c}: predicted {int(pred_counts[c])} times')

    # Average output spike counts per class across test set
    print('\nAverage output spike counts per class (per-output neuron):')
    for c in range(num_classes):
        cnt = per_class_counts[c]
        if cnt > 0:
            avg = per_class_output_sum[c] / float(cnt)
        else:
            avg = np.zeros(int(output.N))
        print(f'  Class {c}: count={cnt}, avg_per_output={avg.tolist()}')

    # Are both output neurons actively spiking (over test set)?
    total_spikes_per_output = np.zeros(int(output.N), dtype=int)
    # sum across per_class_output_sum
    for c in range(num_classes):
        total_spikes_per_output += per_class_output_sum[c].astype(int)

    active = total_spikes_per_output > 0
    for o in range(int(output.N)):
        print(f'Output neuron {o}: total spikes={int(total_spikes_per_output[o])}, active={bool(active[o])}')

    if np.all(active):
        print('\nBoth output neurons are actively spiking across the test set.')
    else:
        print('\nAt least one output neuron is silent across the test set.')

    # Save summary JSON
    try:
        import json
        summary = {
            'prediction_counts': pred_counts.tolist(),
            'avg_output_spikes_per_class': {str(c): per_class_output_sum[c].tolist() for c in range(num_classes)},
            'per_class_sample_counts': per_class_counts,
            'total_spikes_per_output': total_spikes_per_output.tolist(),
        }
        with open(os.path.join('results', 'temporal_diagnostics.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print('\nSaved temporal diagnostics to results/temporal_diagnostics.json')
    except Exception as e:
        print('Warning: failed to save diagnostics JSON:', e)
