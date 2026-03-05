"""
DREAM Benchmark Visualization.

Generates publication-quality plots for benchmark results.
All plots are saved in PDF and PNG formats for arxiv.org submission.

Usage:
    uv run python tests/benchmarks/visualize.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# Use scientific style
matplotlib.use('Agg')  # Non-interactive backend
plt.style.use('default')

# Professional style for arxiv
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Color palette (colorblind-friendly)
COLORS = {
    'dream': '#2E86AB',      # Blue
    'lstm': '#A23B72',       # Purple
    'transformer': '#F18F01', # Orange
}

MARKERS = {
    'dream': 'o',
    'lstm': 's',
    'transformer': '^',
}


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all benchmark results."""
    results = {}

    # Test 1: Basic ASR
    asr_file = results_dir / 'results_basic_asr.json'
    if asr_file.exists():
        with open(asr_file) as f:
            results['basic_asr'] = json.load(f)

    # Test 2: Speaker Adaptation
    adapt_file = results_dir / 'results_speaker_adaptation.json'
    if adapt_file.exists():
        with open(adapt_file) as f:
            results['speaker_adaptation'] = json.load(f)

    # Test 3: Noise Robustness
    noise_file = results_dir / 'results_noise_robustness.json'
    if noise_file.exists():
        with open(noise_file) as f:
            results['noise_robustness'] = json.load(f)

    return results


def plot_training_curves(results: Dict, output_dir: Path) -> None:
    """Plot training curves for Test 1."""
    if 'basic_asr' not in results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get models
    models = list(results['basic_asr'].keys())

    # Plot 1: Loss convergence
    ax1 = axes[0]
    for model in models:
        data = results['basic_asr'][model]
        if 'history' in data and 'loss' in data['history']:
            losses = data['history']['loss']
            epochs = range(1, len(losses) + 1)
            ax1.plot(epochs, losses, label=model.upper(),
                    color=COLORS.get(model, 'gray'),
                    marker=MARKERS.get(model, 'o'),
                    markersize=3,
                    linewidth=2,
                    alpha=0.8)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss (MSE)')
    ax1.set_title('(A) Training Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Final comparison
    ax2 = axes[1]
    model_names = []
    improvements = []
    final_losses = []

    for model in models:
        data = results['basic_asr'][model]
        metrics = data['metrics']
        model_names.append(model.upper())
        improvements.append(metrics['improvement_pct'])
        final_losses.append(metrics['final_loss'])

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, improvements, width,
                    label='Improvement (%)', color=[COLORS.get(m, 'gray') for m in models])
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, final_losses, width,
                         label='Final Loss', color='gray', alpha=0.5)

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Improvement (%)', color=COLORS['dream'])
    ax2_twin.set_ylabel('Final Loss', color='gray')
    ax2.set_title('(B) Final Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_training_curves.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: fig1_training_curves.pdf/png")


def plot_adaptation_results(results: Dict, output_dir: Path) -> None:
    """Plot speaker adaptation results for Test 2."""
    if 'speaker_adaptation' not in results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results['speaker_adaptation'].keys())

    # Plot 1: Adaptation steps
    ax1 = axes[0]
    model_names = []
    steps = []

    for model in models:
        data = results['speaker_adaptation'][model]
        metrics = data['metrics']
        model_names.append(model.upper())
        steps.append(metrics['adaptation_steps'])

    colors = [COLORS.get(m, 'gray') for m in models]
    bars = ax1.bar(model_names, steps, color=colors, edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Adaptation Steps')
    ax1.set_title('(A) Speaker Adaptation Speed')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Spec Target (<50)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, steps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=9)

    # Plot 2: Baseline vs Post-Switch loss
    ax2 = axes[1]
    x = np.arange(len(model_names))
    width = 0.35

    baseline_losses = [results['speaker_adaptation'][m]['metrics']['baseline_loss'] for m in models]
    max_post_losses = [results['speaker_adaptation'][m]['metrics']['max_post_switch'] for m in models]

    bars1 = ax2.bar(x - width/2, baseline_losses, width,
                    label='Baseline Loss', color=[COLORS.get(m, 'gray') for m in models])
    bars2 = ax2.bar(x + width/2, max_post_losses, width,
                    label='Max Post-Switch', color='gray', alpha=0.5)

    ax2.set_ylabel('Reconstruction Loss')
    ax2.set_title('(B) Loss During Speaker Switch')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_speaker_adaptation.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_speaker_adaptation.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: fig2_speaker_adaptation.pdf/png")


def plot_noise_robustness(results: Dict, output_dir: Path) -> None:
    """Plot noise robustness results for Test 3."""
    if 'noise_robustness' not in results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results['noise_robustness'].keys())
    snr_levels = [20, 10, 5, 0]

    # Plot 1: Loss vs SNR
    ax1 = axes[0]
    for model in models:
        data = results['noise_robustness'][model]
        losses = [data['full_results'][snr]['full_loss'] for snr in snr_levels]
        ax1.plot(snr_levels, losses, label=model.upper(),
                color=COLORS.get(model, 'gray'),
                marker=MARKERS.get(model, 'o'),
                markersize=8,
                linewidth=2)

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Reconstruction Loss')
    ax1.set_title('(A) Loss vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Higher SNR on right

    # Plot 2: Loss ratio (10dB / clean)
    ax2 = axes[1]
    model_names = []
    ratios = []

    for model in models:
        data = results['noise_robustness'][model]
        metrics = data['metrics']
        model_names.append(model.upper())
        ratios.append(metrics['loss_ratio'])

    colors = [COLORS.get(m, 'gray') for m in models]
    bars = ax2.bar(model_names, ratios, color=colors, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Loss Ratio (10dB / Clean)')
    ax2.set_title('(B) Noise Sensitivity')
    ax2.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='Stability Threshold (3x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(ratios) * 1.5)

    # Add value labels
    for bar, val in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_noise_robustness.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_noise_robustness.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: fig3_noise_robustness.pdf/png")


def plot_summary_table(results: Dict, output_dir: Path) -> None:
    """Generate summary table as a plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Collect metrics
    table_data = []

    # Test 1: ASR
    if 'basic_asr' in results:
        for model in results['basic_asr']:
            m = results['basic_asr'][model]['metrics']
            passed = '✓' if results['basic_asr'][model]['passed'] else '✗'
            table_data.append([
                model.upper(),
                'ASR Improvement',
                f"{m['improvement_pct']:.1f}%",
                f"{m['final_loss']:.4f}",
                passed
            ])

    # Test 2: Adaptation
    if 'speaker_adaptation' in results:
        for model in results['speaker_adaptation']:
            m = results['speaker_adaptation'][model]['metrics']
            passed = '✓' if results['speaker_adaptation'][model]['passed'] else '✗'
            table_data.append([
                model.upper(),
                'Adaptation Steps',
                str(m['adaptation_steps']),
                f"{m['baseline_loss']:.4f}",
                passed
            ])

    # Test 3: Noise
    if 'noise_robustness' in results:
        for model in results['noise_robustness']:
            m = results['noise_robustness'][model]['metrics']
            passed = '✓' if results['noise_robustness'][model]['passed'] else '✗'
            surprise = 'Yes' if m.get('surprise_responds', False) else 'No'
            table_data.append([
                model.upper(),
                'Noise Robustness',
                f"{m['loss_ratio']:.2f}x",
                surprise,
                passed
            ])

    # Create table
    columns = ['Model', 'Metric', 'Value', 'Details', 'Passed']
    table = ax.table(cellText=table_data, colLabels=columns, loc='center',
                    cellLoc='center', colColors=[COLORS.get(m.lower(), 'gray') for m in results.get('basic_asr', {}).keys()] if table_data else ['gray']*5)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    plt.title('DREAM Benchmark Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'table_summary.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'table_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: table_summary.pdf/png")


def generate_latex_table(results: Dict, output_dir: Path) -> None:
    """Generate LaTeX table for paper."""
    latex = []
    latex.append(r"""
\begin{table}[t]
\centering
\caption{DREAM Benchmark Results: Comparison of DREAM, LSTM, and Transformer models on audio reconstruction tasks.}
\label{tab:benchmark_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{ASR Impr.} & \textbf{Adapt. Steps} & \textbf{Noise Ratio} & \textbf{Passed} \\
\midrule
""")

    models = ['dream', 'lstm', 'transformer']

    for model in models:
        row = [model.upper()]

        # ASR
        if 'basic_asr' in results and model in results['basic_asr']:
            asr = results['basic_asr'][model]['metrics']
            row.append(f"{asr['improvement_pct']:.1f}\\%")
        else:
            row.append('-')

        # Adaptation
        if 'speaker_adaptation' in results and model in results['speaker_adaptation']:
            adapt = results['speaker_adaptation'][model]['metrics']
            row.append(str(adapt['adaptation_steps']))
        else:
            row.append('-')

        # Noise
        if 'noise_robustness' in results and model in results['noise_robustness']:
            noise = results['noise_robustness'][model]['metrics']
            row.append(f"{noise['loss_ratio']:.2f}x")
        else:
            row.append('-')

        # Overall pass
        passed = []
        for test in ['basic_asr', 'speaker_adaptation', 'noise_robustness']:
            if test in results and model in results[test]:
                if results[test][model]['passed']:
                    passed.append('✓')
                else:
                    passed.append('✗')

        row.append(' / '.join(passed) if passed else '-')

        latex.append(' & '.join(row) + r' \\')

    latex.append(r"""
\bottomrule
\end{tabular}
\end{table}
""")

    with open(output_dir / 'benchmark_table.tex', 'w') as f:
        f.write('\n'.join(latex))

    print(f"  ✓ Saved: benchmark_table.tex")


def generate_text_summary(results: Dict, output_dir: Path) -> None:
    """Generate plain text summary."""
    summary = []
    summary.append("=" * 70)
    summary.append("DREAM BENCHMARK RESULTS")
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("=" * 70)
    summary.append("")

    # Test 1
    if 'basic_asr' in results:
        summary.append("TEST 1: Basic ASR Reconstruction")
        summary.append("-" * 40)
        for model, data in results['basic_asr'].items():
            m = data['metrics']
            status = '✓ PASS' if data['passed'] else '✗ FAIL'
            summary.append(f"  {model.upper():12} | Improvement: {m['improvement_pct']:6.1f}% | Final Loss: {m['final_loss']:.6f} | {status}")
        summary.append("")

    # Test 2
    if 'speaker_adaptation' in results:
        summary.append("TEST 2: Speaker Adaptation")
        summary.append("-" * 40)
        for model, data in results['speaker_adaptation'].items():
            m = data['metrics']
            status = '✓ PASS' if data['passed'] else '✗ FAIL'
            summary.append(f"  {model.upper():12} | Adapt Steps: {m['adaptation_steps']:3d} | Baseline: {m['baseline_loss']:.4f} | {status}")
        summary.append("")

    # Test 3
    if 'noise_robustness' in results:
        summary.append("TEST 3: Noise Robustness")
        summary.append("-" * 40)
        for model, data in results['noise_robustness'].items():
            m = data['metrics']
            status = '✓ PASS' if data['passed'] else '✗ FAIL'
            surprise = 'Yes' if m.get('surprise_responds', False) else 'No'
            summary.append(f"  {model.upper():12} | Ratio: {m['loss_ratio']:.2f}x | Surprise: {surprise:3s} | {status}")
        summary.append("")

    # Overall
    summary.append("=" * 70)
    all_passed = all(
        data['passed']
        for test_data in results.values()
        for data in test_data.values()
    )
    summary.append(f"OVERALL: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    summary.append("=" * 70)

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write('\n'.join(summary))

    print(f"  ✓ Saved: summary.txt")


def main():
    """Generate all visualizations."""
    print("=" * 70)
    print("DREAM Benchmark Visualization")
    print("=" * 70)

    # Paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'results'
    output_dir = results_dir / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("\nLoading results...")
    results = load_results(results_dir)

    if not results:
        print("  ✗ No results found. Run benchmarks first:")
        print("    uv run python tests/benchmarks/run_all.py")
        return

    print(f"  ✓ Loaded {len(results)} test results")

    # Generate plots
    print("\nGenerating figures...")

    plot_training_curves(results, output_dir)
    plot_adaptation_results(results, output_dir)
    plot_noise_robustness(results, output_dir)
    plot_summary_table(results, output_dir)

    # Generate tables
    print("\nGenerating tables...")
    generate_latex_table(results, output_dir)
    generate_text_summary(results, output_dir)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles generated:")
    print("  Figures (PDF + PNG):")
    print("    - fig1_training_curves.pdf/png")
    print("    - fig2_speaker_adaptation.pdf/png")
    print("    - fig3_noise_robustness.pdf/png")
    print("    - table_summary.pdf/png")
    print("  Tables:")
    print("    - benchmark_table.tex (LaTeX)")
    print("    - summary.txt (Plain text)")
    print("\nFor arxiv.org submission:")
    print("  1. Include PDF figures in your LaTeX paper")
    print("  2. Copy benchmark_table.tex for results table")
    print("  3. Reference figures as:")
    print("     - Fig. 1: Training convergence comparison")
    print("     - Fig. 2: Speaker adaptation performance")
    print("     - Fig. 3: Noise robustness analysis")
    print("\nExample LaTeX usage:")
    print("  \\usepackage{graphicx}")
    print("  \\includegraphics[width=\\linewidth]{fig1_training_curves.pdf}")
    print("  \\input{benchmark_table.tex}")


if __name__ == '__main__':
    main()
