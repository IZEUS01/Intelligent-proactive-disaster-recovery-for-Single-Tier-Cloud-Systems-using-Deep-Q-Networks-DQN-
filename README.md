# IPDR-Cloud — Intelligent Predictive Disaster Recovery

A Deep Q-Network (DQN) Reinforcement Learning system for cloud disaster recovery that dynamically learns optimal backup scheduling policies from non-stationary workloads. Replaces static Random Forest classification with a true RL agent that adapts to changing cloud conditions in real time.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Flask Backend (app.py)                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │  RL Env       │  │  DQN Agent   │  │  JSON API     │ │
│  │  (rl_env.py)  │──│ (dqn_agent)  │──│  /api/state   │ │
│  │  8 features   │  │  PyTorch NN  │  │  /api/control │ │
│  └──────────────┘  └──────────────┘  └───────┬───────┘ │
└──────────────────────────────────────────────┼─────────┘
                                               │ poll 1s
                                    ┌──────────▼──────────┐
                                    │  Dashboard (HTML/JS) │
                                    │  Chart.js + GSAP     │
                                    └─────────────────────┘
```

## State Space (8 Features)

| Feature | Unit | Description |
|---|---|---|
| `cpu_util` | % | CPU utilization |
| `mem_util` | % | Memory utilization |
| `disk_io` | MB/s | Disk I/O throughput |
| `net_latency` | ms | Network latency |
| `error_rate` | % | Error rate |
| `backup_age` | min | Time since last backup |
| `sla_breach_risk` | 0–1 | Composite breach risk score |
| `workload_trend` | -1 to 1 | Workload slope |

## Action Space

| Action | Label | Backup Interval |
|---|---|---|
| 0 | NORMAL | Every 30 steps |
| 1 | WARNING | Every 10 steps |
| 2 | CRITICAL | Every 2 steps |

## Reward Function

```
R(t) = -10    if SLA breached at failure event
        +1    if failure occurs and SLA is met
       -0.1   per backup triggered
       +0.01  per idle step (no failure, no backup)
```

SLA compliance: **RTO ≤ 5.0 min** and **RPO ≤ 15.0 min**

## DQN Agent

- **Network**: 8 → 64 → 64 → 32 → 3 (ReLU activations)
- **Replay Buffer**: 10,000 transitions
- **Epsilon Decay**: 0.995 per episode (not per step)
- **Target Network**: hard update every 10 episodes
- **Optimizer**: Adam (lr=1e-3)
- **Framework**: PyTorch

## File Structure

```
.
├── docs/                                          # Research papers, reports, trackers, and presentations
│   ├── CY315_Wireless_Mobile_Security_Project_Overview.pdf
│   ├── Disaster_Recovery_Comprehensive_Report.pdf
│   ├── IPDR_Cloud_Presentation.pptx
│   ├── IPDR_Cloud_Research_Paper.pdf
│   └── Research_Paper_Tracker.xlsx
├── data/                                          # Datasets (CSV files)
│   ├── borg_traces_data.csv                       # Google Cluster Trace dataset (Git-ignored due to size)
│   └── processed_trace.csv                        # Preprocessed trace used by Gymnasium env
├── model/                                         # Trained models
│   └── dqn_cloud_dr.pth                           # Saved DQN network weights
├── results/                                       # Experiment metrics and generated charts
│   ├── graph_data.json
│   ├── dqn_rewards.png
│   └── dqn_loss.png
├── templates/                                     # Web interfaces
│   └── index.html                                 # Simulation Dashboard
├── explanations/                                  # Component walkthroughs and mathematical details
│   ├── dqn_agent_explanation.md
│   ├── rl_env_explanation.md
│   ├── run_experiments_explanation.md
│   └── train_dqn_explanation.md
├── app.py                                         # Flask backend & dashboard simulator
├── dqn_agent.py                                   # Deep Q-Network Agent
├── rl_env.py                                      # Custom Gymnasium environment
├── train_dqn.py                                   # Agent training script
├── run_experiments.py                             # Evaluation & benchmark experiments
├── generate_graphs.py                             # Plots result graphs from json data
├── preprocess.py                                  # Preprocessing raw CSVs
├── requirements.txt                               # PyPI dependencies
├── .gitignore                                     # Files excluded from git tracking
└── README.md                                      # Project documentation
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the DQN agent

```bash
python train_dqn.py
```

Trains for 500 episodes, saves model to `model/dqn_cloud_dr.pth`, prints RTO/RPO/SLA compliance results.

### 2. Run full experiments (all 8 graphs)

```bash
python run_experiments.py
```

Runs:
- DQN training (150 episodes)
- Protocol comparison: Reactive vs Rule-Based vs DQN
- Non-stationary adaptation (DQN vs Paper 10 baseline)
- Reward weight ablation (3 configs)
- Feature correlation analysis

Outputs `results/graph_data.json` with data for all 8 research graphs.

### 3. Generate graphs

```bash
python generate_graphs.py
```

Reads `graph_data.json` and produces 8 publication-quality PNG graphs in `results/`.

### 4. Launch live dashboard

```bash
python app.py
```

Opens at `http://localhost:5000`. Features:
- Live telemetry bars (8 features, color-coded thresholds)
- Non-stationary vs stationary trace comparison charts
- DQN decision panel with Q-values
- Backup event log
- Simulation controls (Force Failure, Traffic Spike, Pause/Play)

## Experiment Graphs

| # | Graph | Source |
|---|---|---|
| 1 | SLA Compliance Comparison | Reactive / Rule-Based / DQN |
| 2 | RTO/RPO Reduction | Mean RTO and RPO per protocol |
| 3 | DQN Learning Curve | Cumulative reward per episode |
| 4 | Non-Stationary Adaptation | Sliding-window SLA with workload shift |
| 5 | Backup Action Distribution | Action timeline with failure markers |
| 6 | Reward Convergence | Average reward per episode |
| 7 | Reward Weight Optimization | Ablation across 3 reward configs |
| 8 | Feature Correlation | Pearson correlation with SLA breach |

## Requirements

- Python 3.10+
- PyTorch
- Gymnasium
- Flask
- NumPy, Pandas, Matplotlib
