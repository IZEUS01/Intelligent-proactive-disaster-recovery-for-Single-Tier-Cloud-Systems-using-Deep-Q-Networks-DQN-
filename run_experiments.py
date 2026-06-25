"""
run_experiments.py — Collect all data for 8 graphs, output results/graph_data.json.
No plots. No windows. Just data.
"""
import numpy as np
import random
import json
import os
import pandas as pd

from rl_env import CloudDisasterRecoveryEnv
from dqn_agent import DQNAgent

RESULTS_DIR = "results"
TRAIN_EPISODES = 150
EVAL_STEPS = 2000
SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Baseline Policies (calibrated against Borg trace statistics) ─────────────
# State is now NORMALISED [0,1]. Borg percentiles mapped:
#   error_rate: p50=0.20  p75=0.30  p90=0.35   (normalised /100)
#   cpu_util:   p75=0.09  p90=0.27  p99=0.99   (normalised /15)
#   DQN sees backup_age[5] and sla_breach_risk[6] — baselines do NOT

_last_failure = False

def reactive_policy(state):
    """
    Reactive (worst): stays NORMAL always.
    Fires CRITICAL only the step AFTER a failure — already too late.
    By then backup_age is high and RPO is breached.
    """
    global _last_failure
    action = 2 if _last_failure else 0
    _last_failure = (state[4] > 0.60)   # error_rate >60% = post-failure spike
    return action


def rule_based_policy(state):
    """
    Rule-Based (moderate): fixed thresholds on error_rate + cpu.
    Calibrated to Borg p75/p90. Does NOT use backup_age or sla_breach_risk.
    Cannot anticipate — only sees current stress, not trajectory.
    """
    error = state[4]   # normalised error_rate  (1.0 = 100%)
    cpu   = state[0]   # normalised cpu_util    (1.0 = 15%)
    if error > 0.30 or cpu > 0.27:   # Borg p75 error or p90 cpu
        return 2
    if error > 0.22 or cpu > 0.09:   # Borg p50 error or p75 cpu
        return 1
    return 0


def paper10_fixed_policy(state):
    """
    Paper 10: stationary-trained RL baseline.
    Uses cpu+mem only, thresholds tuned for stationary data.
    Degrades under non-stationary workload shifts.
    """
    cpu, mem = state[0], state[1]
    if cpu > 0.27 or mem > 0.27:
        return 2
    if cpu > 0.09 or mem > 0.13:
        return 1
    return 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def train_agent(env, episodes, verbose=True, label="DQN"):
    """Train a DQN agent. Return agent, cumulative_rewards, avg_rewards."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    cumulative_rewards = []
    avg_rewards = []
    batch_size = 64
    target_update = 10

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        step_rewards = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step(batch_size)
            state = next_state
            total_reward += reward
            step_rewards.append(reward)

        agent.decay_epsilon()
        if ep % target_update == 0:
            agent.update_target_network()

        cumulative_rewards.append(total_reward)
        avg_rewards.append(float(np.mean(step_rewards)))

        if verbose and (ep + 1) % 50 == 0:
            print(f"  [{label}] Episode {ep+1:>4}/{episodes} | "
                  f"CumReward: {total_reward:>8.1f} | "
                  f"AvgReward: {avg_rewards[-1]:>7.3f} | "
                  f"Eps: {agent.epsilon:.3f}")

    return agent, cumulative_rewards, avg_rewards


def evaluate_policy(policy_fn, steps=EVAL_STEPS, env_kwargs=None):
    """Run a policy for N steps. Return metrics dict and the env."""
    global _last_failure
    _last_failure = False   # reset reactive state between runs
    random.seed(SEED)
    np.random.seed(SEED)
    kw = env_kwargs or {}
    env = CloudDisasterRecoveryEnv(max_steps=steps, **kw)
    state, _ = env.reset()
    done = False
    while not done:
        action = policy_fn(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_metrics(), env


def evaluate_agent(agent, steps=EVAL_STEPS, env_kwargs=None):
    """Run the DQN agent for N steps. Return metrics dict and the env."""
    random.seed(SEED)
    np.random.seed(SEED)
    kw = env_kwargs or {}
    env = CloudDisasterRecoveryEnv(max_steps=steps, **kw)
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_metrics(), env


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    output = {}

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Train DQN → Graphs 3, 6
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  STEP 1: Training DQN agent")
    print("=" * 60)

    env = CloudDisasterRecoveryEnv()
    agent, cumulative_rewards, avg_rewards = train_agent(
        env, TRAIN_EPISODES, verbose=True, label="Main-DQN"
    )
    os.makedirs("model", exist_ok=True)
    agent.save("model/dqn_cloud_dr.pth")
    print(f"  Model saved.\n")

    output["graph3"] = {
        "learning_curve": cumulative_rewards,
        "total_episodes": TRAIN_EPISODES
    }
    output["graph6"] = {
        "avg_reward_per_episode": avg_rewards,
        "smoothing_window": 50
    }

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Evaluate 3 protocols → Graphs 1, 2
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  STEP 2: Protocol comparison (Reactive / Rule-Based / DQN)")
    print("=" * 60)

    reactive_m, _ = evaluate_policy(reactive_policy)
    rule_m, _ = evaluate_policy(rule_based_policy)
    dqn_m, dqn_eval_env = evaluate_agent(agent)

    print(f"  {'Protocol':<15} {'SLA%':>8} {'RTO':>8} {'RPO':>8} {'Breaches':>10}")
    print(f"  {'-'*49}")
    for name, m in [("Reactive", reactive_m), ("Rule-Based", rule_m), ("DQN-Ours", dqn_m)]:
        print(f"  {name:<15} {m['sla_compliance_rate']:>7.1f}% "
              f"{m['mean_rto_min']:>7.1f}m {m['mean_rpo_min']:>7.1f}m "
              f"{m['total_sla_breaches']:>10d}")
    print()

    output["graph1"] = {
        "reactive_sla": reactive_m["sla_compliance_rate"],
        "rulebased_sla": rule_m["sla_compliance_rate"],
        "dqn_sla": dqn_m["sla_compliance_rate"]
    }
    output["graph2"] = {
        "reactive_rto": reactive_m["mean_rto_min"],
        "reactive_rpo": reactive_m["mean_rpo_min"],
        "rulebased_rto": rule_m["mean_rto_min"],
        "rulebased_rpo": rule_m["mean_rpo_min"],
        "dqn_rto": dqn_m["mean_rto_min"],
        "dqn_rpo": dqn_m["mean_rpo_min"]
    }

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Action distribution from DQN eval → Graph 5
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  STEP 3: Extracting action distribution (Graph 5)")
    print("=" * 60)

    action_log = dqn_eval_env.full_action_log
    output["graph5"] = {
        "action_steps": [a["step"] for a in action_log],
        "actions": [a["action"] for a in action_log],
        "failure_steps": [a["step"] for a in action_log if a["is_failure"]]
    }
    print(f"  Logged {len(action_log)} steps, "
          f"{len(output['graph5']['failure_steps'])} failure events.\n")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Non-stationary adaptation → Graph 4
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  STEP 4: Non-stationary adaptation (DQN vs Paper 10)")
    print("=" * 60)

    shift_step = 1000
    total_sim_steps = 3000
    window = 500

    graph4_data = {"workload_shift_step": shift_step}

    for tag, policy_fn in [("dqn_sliding_sla", None), ("paper10_sliding_sla", paper10_fixed_policy)]:
        random.seed(SEED)
        np.random.seed(SEED)
        sim_env = CloudDisasterRecoveryEnv(max_steps=total_sim_steps)
        sim_env.enable_workload_shift(shift_step)
        state, _ = sim_env.reset()
        done = False

        sla_window = []
        sla_over_time = []
        time_indices = []
        step = 0

        while not done:
            if policy_fn is None:
                action = agent.select_action(state, evaluate=True)
            else:
                action = policy_fn(state)
            state, reward, terminated, truncated, info = sim_env.step(action)
            done = terminated or truncated

            sla_met = 1 if (sim_env.rto <= 5.0 and sim_env.rpo <= 15.0) else 0
            sla_window.append(sla_met)
            if len(sla_window) > window:
                sla_window.pop(0)

            if step % 10 == 0:
                sla_over_time.append(round((sum(sla_window) / len(sla_window)) * 100, 2))
                time_indices.append(step)
            step += 1

        graph4_data[tag] = sla_over_time
        if "time_steps" not in graph4_data:
            graph4_data["time_steps"] = time_indices

        label = "DQN" if policy_fn is None else "Paper10"
        print(f"  {label} final sliding SLA: {sla_over_time[-1]:.1f}%")

    output["graph4"] = graph4_data
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Reward weight ablation → Graph 7
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  STEP 5: Reward weight ablation (3 configs)")
    print("=" * 60)

    ablation_configs = [
        {"label": "Conservative", "breach_penalty": -20.0, "survival_bonus": 25.0,
         "stability_reward": 2.0, "backup_cost": -0.05, "unnecessary_critical_cost": -0.5},
        {"label": "Balanced", "breach_penalty": -10.0, "survival_bonus": 15.0,
         "stability_reward": 1.0, "backup_cost": -0.10, "unnecessary_critical_cost": -2.0},
        {"label": "Aggressive", "breach_penalty": -5.0, "survival_bonus": 10.0,
         "stability_reward": 0.5, "backup_cost": -0.50, "unnecessary_critical_cost": -5.0},
    ]

    graph7_configs = []
    for cfg in ablation_configs:
        label = cfg["label"]
        reward_config = {k: v for k, v in cfg.items() if k != "label"}
        print(f"  Training {label} (80 eps)...")

        abl_env = CloudDisasterRecoveryEnv(reward_config=reward_config)
        abl_agent, _, _ = train_agent(abl_env, 80, verbose=False, label=label)

        abl_m, _ = evaluate_agent(abl_agent)
        graph7_configs.append({
            "label": label,
            "breach_penalty": cfg["breach_penalty"],
            "backup_cost": cfg["backup_cost"],
            "sla_compliance": abl_m["sla_compliance_rate"],
            "mean_rto_min": abl_m["mean_rto_min"]
        })
        print(f"    {label}: SLA={abl_m['sla_compliance_rate']:.1f}%, "
              f"RTO={abl_m['mean_rto_min']:.1f}m")

    output["graph7"] = {"configs": graph7_configs}
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Feature correlation → Graph 8
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  STEP 6: Feature correlation with SLA breach (Graph 8)")
    print("=" * 60)

    feature_names = ["cpu_util", "mem_util", "disk_io", "net_latency",
                     "error_rate", "backup_age", "sla_breach_risk", "workload_trend"]

    df = pd.DataFrame(dqn_eval_env.full_state_log)
    df["sla_breach"] = df["sla_breach"].astype(float)

    correlations = []
    for feat in feature_names:
        c = abs(df[feat].corr(df["sla_breach"]))
        correlations.append(round(c if not np.isnan(c) else 0.0, 4))
        print(f"  {feat:<20} {correlations[-1]:.4f}")

    output["graph8"] = {
        "feature_names": feature_names,
        "correlations": correlations
    }
    print()

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════════
    out_path = os.path.join(RESULTS_DIR, "graph_data.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print(f"  GRAPH DATA READY")
    print(f"  Saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
