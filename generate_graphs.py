"""
generate_graphs.py — Reads results/graph_data.json and produces all 8 publication-quality graphs.

Outputs: results/graph1_sla_compliance.png ... results/graph8_feature_correlation.png
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = "results"
DATA_PATH = os.path.join(RESULTS_DIR, "graph_data.json")

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor': '#141414',
    'axes.edgecolor': '#333',
    'axes.labelcolor': '#ccc',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'text.color': '#ccc',
    'grid.color': '#222',
    'grid.linewidth': 0.5,
    'font.family': 'monospace',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3
})

COLORS = {
    'blue': '#3b82f6',
    'purple': '#a855f7',
    'green': '#22c55e',
    'red': '#ef4444',
    'amber': '#f59e0b',
    'cyan': '#06b6d4',
    'pink': '#ec4899',
    'white': '#e5e5e5'
}


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Graph 1: SLA Compliance Comparison (Grouped Bar)
# ─────────────────────────────────────────────────────────────────────────────
def graph1(data):
    d = data['graph1']
    display_map = {
        'reactive_sla': 'Reactive',
        'rulebased_sla': 'Rule-Based',
        'dqn_sla': 'DQN-Ours'
    }
    labels = [display_map.get(k, k) for k in d.keys()]
    values = list(d.values())
    colors = [COLORS['red'], COLORS['amber'], COLORS['green']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='#333')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('SLA Compliance (%)')
    ax.set_title('Graph 1 — SLA Compliance Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=95, color=COLORS['cyan'], linestyle='--', alpha=0.5, label='95% Target')
    ax.legend(loc='lower right')
    
    save(fig, 'graph1_sla_compliance.png')


# ─────────────────────────────────────────────────────────────────────────────
# Graph 2: RTO/RPO Reduction (Grouped Bar)
# ─────────────────────────────────────────────────────────────────────────────
def graph2(data):
    d = data['graph2']
    labels = ['Reactive', 'Rule-Based', 'DQN-Ours']
    rto_vals = [d['reactive_rto'], d['rulebased_rto'], d['dqn_rto']]
    rpo_vals = [d['reactive_rpo'], d['rulebased_rpo'], d['dqn_rpo']]
    
    x = np.arange(len(labels))
    width = 0.3
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, rto_vals, width, label='Mean RTO (min)', color=COLORS['blue'], edgecolor='#333')
    bars2 = ax.bar(x + width/2, rpo_vals, width, label='Mean RPO (min)', color=COLORS['purple'], edgecolor='#333')
    
    for bar, val in zip(bars1, rto_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, rpo_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Minutes')
    ax.set_title('Graph 2 — RTO/RPO Reduction Across Protocols', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=5, color=COLORS['cyan'], linestyle='--', alpha=0.4, label='RTO Target')
    ax.axhline(y=15, color=COLORS['pink'], linestyle='--', alpha=0.4, label='RPO Target')
    
    save(fig, 'graph2_rto_rpo.png')


# ─────────────────────────────────────────────────────────────────────────────
# Graph 3: DQN Learning Curve (Cumulative Reward per Episode)
# ─────────────────────────────────────────────────────────────────────────────
def graph3(data):
    d = data['graph3']
    rewards = d['learning_curve']
    episodes = list(range(1, len(rewards) + 1))
    
    # Smoothed line
    window = max(1, len(rewards) // 20)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, color=COLORS['blue'], alpha=0.3, linewidth=0.5, label='Raw')
    ax.plot(episodes[:len(smoothed)], smoothed, color=COLORS['cyan'], linewidth=2, label=f'Smoothed (w={window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Graph 3 — DQN Learning Curve', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save(fig, 'graph3_learning_curve.png')


# ─────────────────────────────────────────────────────────────────────────────
# Graph 4: Non-Stationary Adaptation
# ─────────────────────────────────────────────────────────────────────────────
def graph4(data):
    d = data['graph4']
    shift = d['workload_shift_step']
    steps = d['time_steps']
    dqn_sla = d['dqn_sliding_sla']
    p10_sla = d['paper10_sliding_sla']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(steps, dqn_sla, color=COLORS['green'], linewidth=2, label='DQN-Ours')
    ax.plot(steps, p10_sla, color=COLORS['red'], linewidth=2, label='Paper10 Baseline', linestyle='--')
    
    ax.axvline(x=shift, color=COLORS['amber'], linestyle='--', linewidth=1.5, alpha=0.8, label=f'Workload Shift (step {shift})')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('SLA Compliance (%, sliding window)')
    ax.set_title('Graph 4 — Non-Stationary Adaptation', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    
    save(fig, 'graph4_nonstationary.png')


# ─────────────────────────────────────────────────────────────────────────────
# Graph 5: Backup Action Distribution Over Time
# ─────────────────────────────────────────────────────────────────────────────
def graph5(data):
    d = data['graph5']
    steps = d['action_steps']
    actions = d['actions']
    fail_steps = d['failure_steps']
    
    action_colors = {0: COLORS['green'], 1: COLORS['amber'], 2: COLORS['red']}
    action_labels = {0: 'NORMAL', 1: 'WARNING', 2: 'CRITICAL'}
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot actions as colored scatter
    for a_val in [0, 1, 2]:
        mask = [i for i, a in enumerate(actions) if a == a_val]
        ax.scatter([steps[i] for i in mask], [actions[i] for i in mask],
                   color=action_colors[a_val], s=2, alpha=0.6, label=action_labels[a_val])
    
    # Failure markers
    for fs in fail_steps:
        ax.axvline(x=fs, color=COLORS['red'], alpha=0.3, linewidth=0.5)
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['NORMAL', 'WARNING', 'CRITICAL'])
    ax.set_xlabel('Time Step')
    ax.set_title('Graph 5 — Backup Action Distribution Over Time', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', markerscale=5)
    ax.grid(axis='x', alpha=0.2)
    
    save(fig, 'graph5_action_distribution.png')


# ─────────────────────────────────────────────────────────────────────────────
# Graph 6: Reward Convergence (Avg Reward per Episode)
# ─────────────────────────────────────────────────────────────────────────────
def graph6(data):
    d = data['graph6']
    rewards = d['avg_reward_per_episode']
    episodes = list(range(1, len(rewards) + 1))
    
    window = max(1, len(rewards) // 20)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, color=COLORS['purple'], alpha=0.3, linewidth=0.5, label='Raw')
    ax.plot(episodes[:len(smoothed)], smoothed, color=COLORS['pink'], linewidth=2, label=f'Smoothed (w={window})')
    
    ax.axhline(y=0, color='#555', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward per Step')
    ax.set_title('Graph 6 — Reward Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save(fig, 'graph6_reward_convergence.png')


# ─────────────────────────────────────────────────────────────────────────────
# Graph 7: Reward Weight Optimization (Ablation)
# ─────────────────────────────────────────────────────────────────────────────
def graph7(data):
    d = data['graph7']
    configs = d['configs']
    labels = [c['label'] for c in configs]
    sla_vals = [c['sla_compliance'] for c in configs]
    rto_vals = [c['mean_rto_min'] for c in configs]
    
    # Create labels with penalty info
    display_labels = []
    for c in configs:
        bp = c['breach_penalty']
        bc = c['backup_cost']
        display_labels.append(f"{c['label']}\n(breach={bp}, cost={bc})")
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, sla_vals, width, label='SLA Compliance (%)', color=COLORS['green'], edgecolor='#333')
    ax1.set_ylabel('SLA Compliance (%)', color=COLORS['green'])
    ax1.set_ylim(0, 105)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, rto_vals, width, label='Mean RTO (min)', color=COLORS['blue'], edgecolor='#333')
    ax2.set_ylabel('Mean RTO (min)', color=COLORS['blue'])
    
    for bar, val in zip(bars1, sla_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color=COLORS['green'])
    for bar, val in zip(bars2, rto_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}m', ha='center', va='bottom', fontsize=9, color=COLORS['blue'])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, fontsize=8)
    ax1.set_title('Graph 7 — Reward Weight Optimization', fontsize=13, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    ax1.grid(axis='y', alpha=0.2)
    
    save(fig, 'graph7_reward_ablation.png')


# ─────────────────────────────────────────────────────────────────────────────
# Graph 8: Feature Correlation with SLA Breach (Horizontal Bar)
# ─────────────────────────────────────────────────────────────────────────────
def graph8(data):
    d = data['graph8']
    features = d['feature_names']
    correlations = d['correlations']
    
    # Sort by absolute correlation
    pairs = sorted(zip(features, correlations), key=lambda x: abs(x[1]))
    features_sorted = [p[0] for p in pairs]
    corr_sorted = [p[1] for p in pairs]
    
    colors = [COLORS['red'] if c > 0 else COLORS['blue'] for c in corr_sorted]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features_sorted, corr_sorted, color=colors, edgecolor='#333', height=0.6)
    
    for bar, val in zip(bars, corr_sorted):
        offset = 0.02 if val >= 0 else -0.02
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha=ha, va='center', fontsize=9)
    
    ax.axvline(x=0, color='#555', linewidth=0.8)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Graph 8 — Feature Correlation with SLA Breach', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    save(fig, 'graph8_feature_correlation.png')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\nLoading experiment data from {DATA_PATH}...")
    data = load_data()
    
    print("Generating graphs...\n")
    graph1(data)
    graph2(data)
    graph3(data)
    graph4(data)
    graph5(data)
    graph6(data)
    graph7(data)
    graph8(data)
    
    print(f"\nAll 8 graphs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
