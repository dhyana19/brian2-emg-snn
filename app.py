import argparse
import subprocess
import sys


DEFAULT_MODE = "baseline"


def build_command(
    mode,
    label_mode,
    hidden,
    train_samples,
    test_samples,
    time_bins,
    feature_mode,
    encoder_mode,
    n_train,
    results_dir,
    curve_train_sizes,
    curve_feature_modes,
):
    python = sys.executable

    if mode == "baseline":
        return [
            python,
            "src/baseline.py",
            "--label-mode",
            label_mode,
            "--results-dir",
            results_dir,
        ]

    if mode == "hidden_readout":
        cmd = [
            python,
            "src/eval_snn_hidden_readout.py",
            "--label-mode",
            label_mode,
            "--hidden",
            str(hidden),
            "--time-bins",
            str(time_bins),
            "--feature-mode",
            feature_mode,
            "--encoder-mode",
            encoder_mode,
            "--results-dir",
            results_dir,
        ]
        if train_samples is not None:
            cmd.extend(["--train-samples", str(train_samples)])
        if test_samples is not None:
            cmd.extend(["--test-samples", str(test_samples)])
        return cmd

    if mode == "snn_output_wta_eval":
        cmd = [
            python,
            "src/eval_snn.py",
            "--label-mode",
            label_mode,
            "--hidden",
            str(hidden),
            "--readout-mode",
            "output_wta",
            "--results-dir",
            results_dir,
        ]
        if test_samples is not None:
            cmd.extend(["--test-samples", str(test_samples)])
        return cmd

    if mode == "snn_prototype_train":
        cmd = [
            python,
            "src/train_snn.py",
            "--label-mode",
            label_mode,
            "--hidden",
            str(hidden),
            "--readout-mode",
            "hidden_temporal_prototype",
            "--time-bins",
            str(time_bins),
            "--encoder-mode",
            encoder_mode,
            "--n-train",
            str(n_train),
            "--results-dir",
            results_dir,
        ]
        return cmd

    if mode == "snn_prototype_eval":
        cmd = [
            python,
            "src/eval_snn.py",
            "--label-mode",
            label_mode,
            "--hidden",
            str(hidden),
            "--readout-mode",
            "hidden_temporal_prototype",
            "--time-bins",
            str(time_bins),
            "--encoder-mode",
            encoder_mode,
            "--results-dir",
            results_dir,
        ]
        if test_samples is not None:
            cmd.extend(["--test-samples", str(test_samples)])
        return cmd

    if mode == "deployable_baseline_train":
        return [
            python,
            "src/deployable_baseline.py",
            "--mode",
            "train",
            "--label-mode",
            label_mode,
            "--results-dir",
            results_dir,
        ]

    if mode == "deployable_baseline_eval":
        return [
            python,
            "src/deployable_baseline.py",
            "--mode",
            "eval",
            "--label-mode",
            label_mode,
            "--results-dir",
            results_dir,
        ]

    if mode == "learning_curves":
        return [
            python,
            "src/learning_curves.py",
            "--label-mode",
            label_mode,
            "--feature-modes",
            curve_feature_modes,
            "--train-sizes",
            curve_train_sizes,
            "--test-samples",
            str(test_samples),
            "--hidden",
            str(hidden),
            "--time-bins",
            str(time_bins),
            "--encoder-mode",
            encoder_mode,
            "--results-dir",
            results_dir,
        ]

    raise ValueError(f"Unknown mode: {mode}")


def run_command(
    mode,
    label_mode="rest_2_3",
    hidden=10,
    train_samples=300,
    test_samples=100,
    time_bins=5,
    feature_mode="hidden",
    encoder_mode="envelope",
    n_train=300,
    results_dir="results",
    curve_train_sizes="250,500,1000,2000,3000",
    curve_feature_modes="handcrafted,hybrid",
):
    cmd = build_command(
        mode=mode,
        label_mode=label_mode,
        hidden=hidden,
        train_samples=train_samples,
        test_samples=test_samples,
        time_bins=time_bins,
        feature_mode=feature_mode,
        encoder_mode=encoder_mode,
        n_train=n_train,
        results_dir=results_dir,
        curve_train_sizes=curve_train_sizes,
        curve_feature_modes=curve_feature_modes,
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    summary = f"$ {' '.join(cmd)}\n\n"
    if result.stdout:
        summary += result.stdout
    if result.stderr:
        summary += "\n[stderr]\n" + result.stderr
    if result.returncode != 0:
        summary += f"\nProcess exited with code {result.returncode}\n"
    return summary


def launch_demo():
    import gradio as gr

    demo = gr.Interface(
        fn=run_command,
        inputs=[
            gr.Dropdown(
                choices=[
                    "baseline",
                    "hidden_readout",
                    "snn_output_wta_eval",
                    "snn_prototype_train",
                    "snn_prototype_eval",
                    "deployable_baseline_train",
                    "deployable_baseline_eval",
                    "learning_curves",
                ],
                value=DEFAULT_MODE,
                label="Mode",
            ),
            gr.Dropdown(
                choices=["binary", "rest_2_3", "full_6", "full_7_optional"],
                value="rest_2_3",
                label="Label Mode",
            ),
            gr.Number(value=10, precision=0, label="Hidden Units"),
            gr.Number(value=300, precision=0, label="Train Samples"),
            gr.Number(value=100, precision=0, label="Test Samples"),
            gr.Number(value=5, precision=0, label="Time Bins"),
            gr.Dropdown(
                choices=["hidden", "handcrafted", "hybrid"],
                value="hidden",
                label="Feature Mode",
            ),
            gr.Dropdown(
                choices=["envelope", "envelope_delta"],
                value="envelope",
                label="Encoder Mode",
            ),
            gr.Number(value=300, precision=0, label="N Train"),
            gr.Textbox(value="results", label="Results Dir"),
            gr.Textbox(value="250,500,1000,2000,3000", label="Curve Train Sizes"),
            gr.Textbox(value="handcrafted,hybrid", label="Curve Feature Modes"),
        ],
        outputs=gr.Textbox(label="Logs", lines=30),
        title="EMG Gesture Experiment Runner",
        description=(
            "Thin launcher for baseline, hidden-readout, and SNN prototype experiments. "
            "Use rest_2_3 for the multiclass path."
        ),
    )

    demo.launch()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launcher for EMG gesture experiments."
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=[
            "baseline",
            "hidden_readout",
            "snn_output_wta_eval",
            "snn_prototype_train",
            "snn_prototype_eval",
            "deployable_baseline_train",
            "deployable_baseline_eval",
            "learning_curves",
        ],
        help="Run one experiment directly in the terminal. If omitted, launch the Gradio UI.",
    )
    parser.add_argument("--label-mode", default="rest_2_3")
    parser.add_argument("--hidden", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=300)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--time-bins", type=int, default=5)
    parser.add_argument("--feature-mode", default="hidden")
    parser.add_argument("--encoder-mode", default="envelope")
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--curve-train-sizes", default="250,500,1000,2000,3000")
    parser.add_argument("--curve-feature-modes", default="handcrafted,hybrid")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode is None:
        launch_demo()
    else:
        print(
            run_command(
                mode=args.mode,
                label_mode=args.label_mode,
                hidden=args.hidden,
                train_samples=args.train_samples,
                test_samples=args.test_samples,
                time_bins=args.time_bins,
                feature_mode=args.feature_mode,
                encoder_mode=args.encoder_mode,
                n_train=args.n_train,
                results_dir=args.results_dir,
                curve_train_sizes=args.curve_train_sizes,
                curve_feature_modes=args.curve_feature_modes,
            )
        )
