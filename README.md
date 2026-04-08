# EMG Gesture Classification using Brian2 SNNs

A complete EMG gesture classification pipeline developed as part of 
an internship project at Carleton University. The project implements 
and compares three approaches: a handcrafted feature baseline, a 
hybrid SNN, and a pure Brian2 spiking neural network, evaluated 
under strict subject-wise held-out testing across binary, 3-class, 
and 6-class gesture tasks.

## Results Summary

| Model            | Task    | Accuracy | Macro F1 |
|------------------|---------|----------|----------|
| Handcrafted LR   | Binary  | 98.95%   | 0.9816   |
| Handcrafted LR   | 3-class | 93.20%   | 0.9312   |
| Handcrafted LR   | 6-class | 84.09%   | 0.8401   |
| Hybrid SNN       | 3-class | 93.25%   | 0.9316   |
| Hybrid SNN       | 6-class | 84.11%   | 0.8404   |
| Pure Brian2 SNN  | Binary  | 84.73%   | 0.7191   |
| Pure Brian2 SNN  | 6-class | 23.79%   | 0.1384   |
| SpikingJelly SNN | 6-class | 85.00%   | 0.8497   |

## Project Structure
brian2_emg_project/
├── src/
│   ├── snn.py                      # Brian2 SNN architecture
│   ├── train_snn.py                # SNN training pipeline
│   ├── eval_snn.py                 # SNN evaluation pipeline
│   ├── windowing.py                # EMG windowing and encoding
│   ├── build_dataset.py            # Dataset loading
│   ├── label_modes.py              # Gesture label mappings
│   ├── emg_features.py             # Handcrafted feature extraction
│   ├── baseline.py                 # Baseline model
│   ├── hidden_temporal_readout.py  # Hidden spike readout
│   ├── experiment_logging.py       # Result logging utilities
│   └── learning_curves.py         # Learning curve generation
├── experiments/
│   ├── common.py                   # Shared utilities
│   ├── exp1_baseline_3class.py     # LR baseline, 3-class
│   ├── exp2_baseline_6class.py     # LR baseline, 6-class
│   ├── exp3_baseline_binary.py     # LR baseline, binary
│   ├── exp4_hybrid_3class.py       # Hybrid SNN, 3-class
│   ├── exp5_snn_binary.py          # Pure Brian2 SNN, binary
│   ├── exp6_snn_6class.py          # Pure Brian2 SNN, 6-class
│   ├── exp7_learning_curves.py     # Learning curves
│   ├── exp8_hybrid_6class.py       # Hybrid SNN, 6-class
│   └── exp10_spikingjelly_6class.py # SpikingJelly surrogate SNN
├── .gitignore
├── requirements.txt
└── README.md

## Dataset

UCI EMG Dataset for Gestures:
https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures

Download and place in:
data/EMG_data_for_gestures-master/

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/brian2-emg-snn.git
cd brian2-emg-snn
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Running Experiments

Each script in `experiments/` is fully standalone and reproducible.
Run from the project root:

```bash
# Handcrafted baselines
python experiments/exp1_baseline_3class.py
python experiments/exp2_baseline_6class.py
python experiments/exp3_baseline_binary.py

# Hybrid SNN
python experiments/exp4_hybrid_3class.py
python experiments/exp8_hybrid_6class.py

# Pure Brian2 SNN
python experiments/exp5_snn_binary.py
python experiments/exp6_snn_6class.py

# SpikingJelly surrogate gradient SNN
python experiments/exp10_spikingjelly_6class.py

# Learning curves
python experiments/exp7_learning_curves.py
```

Results are saved to the `results/` folder as JSON files,
confusion matrix PNGs, and training curve plots.

## Key Findings

- Handcrafted LR baseline is strong and stable across all tasks
- Pure Brian2 SNN achieves 84.73% on binary task with 
  genuine output-spike decisions
- Multiclass SNN collapses due to output neuron symmetry,
  documented with epoch-level diagnostics
- SpikingJelly surrogate gradient SNN achieves 85.00% on 
  6-class, confirming the architecture is viable and the 
  Brian2 training rule was the bottleneck

## Author

Dhyana Chandravadan Parmar (22BIT099)
Carleton University Internship, Project A
Supervisor: Prof. Leonard