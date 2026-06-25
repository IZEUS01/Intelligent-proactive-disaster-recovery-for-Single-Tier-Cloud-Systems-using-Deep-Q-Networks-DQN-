# `run_experiments.py` — The Examiner (Detailed Explanation)

If `train_dqn.py` is the AI's training camp, `run_experiments.py` is the **scientific research lab**. Its only job is to run massive, automated comparisons between our smart AI and "dumb" traditional methods, collect the results, and save them so we can build research charts.

---

## Part 1: The "Dumb" Competitors (Lines 30–68)

To prove our AI is good, we have to race it against older, traditional ways of taking backups.

```python
def reactive_policy(state):
    """Fires CRITICAL only the step AFTER a failure."""
    global _last_failure
    action = 2 if _last_failure else 0
    _last_failure = (state[4] > 0.60)
    return action

def rule_based_policy(state):
    """Fixed thresholds on error_rate + cpu."""
    error = state[4]
    cpu   = state[0]
    if error > 0.30 or cpu > 0.27:
        return 2
    if error > 0.22 or cpu > 0.09:
        return 1
    return 0
```

**What's happening:**
- **`reactive_policy`**: This strategy is lazy. It does absolutely nothing until a server crashes, and *then* it panics and takes a backup. It always loses data.
- **`rule_based_policy`**: This strategy follows a strict, hard-coded rulebook. "If CPU is exactly above 27%, take a backup." It can't learn, it can't anticipate the future, and it can't adapt if the server traffic changes.

---

## Part 2: The Main Experiment Loop (Lines 150–175)

```python
def main():
    output = {}

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Train DQN → Graphs 3, 6
    # ═══════════════════════════════════════════════════════════════════════
    env = CloudDisasterRecoveryEnv()
    agent, cumulative_rewards, avg_rewards = train_agent(
        env, TRAIN_EPISODES, verbose=True, label="Main-DQN"
    )
    agent.save("model/dqn_cloud_dr.pth")

    output["graph3"] = {"learning_curve": cumulative_rewards, ...}
```

**What's happening:**
The script creates a massive empty dictionary called `output`. It's going to fill this dictionary with the results of 6 different experiments, and then save it to a JSON file.
- **STEP 1** just trains the AI from scratch so we have fresh learning data to put into our first two graphs.

---

## Part 3: Protocol Comparison (Lines 177–208)

```python
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Evaluate 3 protocols → Graphs 1, 2
    # ═══════════════════════════════════════════════════════════════════════
    reactive_m, _ = evaluate_policy(reactive_policy)
    rule_m, _ = evaluate_policy(rule_based_policy)
    dqn_m, dqn_eval_env = evaluate_agent(agent)
```

**What's happening:**
This is the big race. It runs the simulation 3 times:
1. Once using the lazy `reactive_policy`.
2. Once using the strict `rule_based_policy`.
3. Once using our fully trained AI `agent`.

It records the SLA compliance rate (e.g., Reactive got 40%, Rule-Based got 85%, AI got 99%) and saves them into the `output` dictionary for Graphs 1 and 2.

---

## Part 4: Action Distribution (Lines 210–224)

```python
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Action distribution from DQN eval → Graph 5
    # ═══════════════════════════════════════════════════════════════════════
    action_log = dqn_eval_env.full_action_log
    output["graph5"] = {
        "action_steps": [a["step"] for a in action_log],
        "actions": [a["action"] for a in action_log],
        "failure_steps": [a["step"] for a in action_log if a["is_failure"]]
    }
```

**What's happening:**
We want to see *how* the AI is behaving. Is it constantly pressing the CRITICAL button, or is it spreading out its backups? This step pulls the AI's exact click history and logs it so we can draw a timeline of its choices.

---

## Part 5: Non-Stationary Adaptation (Lines 226–278)

```python
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Non-stationary adaptation → Graph 4
    # ═══════════════════════════════════════════════════════════════════════
    shift_step = 1000
    sim_env.enable_workload_shift(shift_step)
```

**What's happening:**
"Non-stationary" is a fancy way of saying "the rules suddenly change." 
This test injects a massive traffic spike at step 1000. It pits our AI against another hard-coded baseline ("Paper 10"). Because the baseline is rigid, its SLA compliance plummets when the traffic spikes. Our AI, however, notices the changing data and adapts its backup strategy to survive the spike.

---

## Part 6: Ablation Study (Lines 280–317)

```python
    ablation_configs = [
        {"label": "Conservative", "breach_penalty": -20.0, "backup_cost": -0.05...},
        {"label": "Balanced", "breach_penalty": -10.0, "backup_cost": -0.10...},
        {"label": "Aggressive", "breach_penalty": -5.0, "backup_cost": -0.50...},
    ]
```

**What's happening:**
An "ablation study" means taking parts of the system away to see how it breaks. Here, we change the AI's personality:
- **Conservative:** Crashing is highly illegal (-20 penalty), backups are extremely cheap (-0.05). The AI becomes paranoid and backs up constantly.
- **Aggressive:** Crashing isn't that bad (-5 penalty), but taking backups is very expensive (-0.50). The AI becomes a risk-taker and rarely backs up.
We train 3 mini-AIs with these different personalities and record how they perform.

---

## Part 7: Feature Correlation (Lines 319–342)

```python
    feature_names = ["cpu_util", "mem_util", "disk_io", "net_latency",
                     "error_rate", "backup_age", "sla_breach_risk", "workload_trend"]

    df = pd.DataFrame(dqn_eval_env.full_state_log)
    correlations = []
    for feat in feature_names:
        c = abs(df[feat].corr(df["sla_breach"]))
        correlations.append(round(c if not np.isnan(c) else 0.0, 4))
```

**What's happening:**
This answers the question: *Which server metrics actually cause crashes?*
It runs a statistical correlation (math formula) comparing each metric against SLA breaches. It usually proves that `error_rate` and `backup_age` are the #1 reasons a server crashes and loses data, proving that our AI is smart to focus on them.

---

## Part 8: Saving the Data (Lines 344–358)

```python
    out_path = os.path.join(RESULTS_DIR, "graph_data.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
```

**What's happening:**
Finally, after running all 6 experiments, it takes the giant `output` dictionary and saves it to `results/graph_data.json`. 

**Why no charts here?** Because drawing charts takes up memory. `run_experiments.py` only gathers the raw math. A separate file (`generate_graphs.py`) is responsible for reading this JSON file and actually drawing the visual lines and bars.
