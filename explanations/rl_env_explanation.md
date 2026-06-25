# `rl_env.py` — Detailed Explanation

This file is the **heart of the entire project**. It creates a virtual simulation of a cloud datacenter that the AI "lives inside" and learns from. Think of it as a **video game world** — the AI is the player, the servers are the game, and crashes are the enemies.

---

## Part 1: Imports & Setup (Lines 1–36)

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd
from collections import deque

TRACE_PATH = "data/processed_trace.csv"
```

**What's happening:**
- `gymnasium` is a popular library for building AI training environments (like game levels for robots).
- `numpy` is used for math operations.
- `pandas` is used for reading and working with the CSV data file.
- `deque` is a special list that automatically throws away old items when it gets too long (like a security camera that only keeps the last 100 frames).
- `TRACE_PATH` tells the code where to find the cleaned-up Google server data.

---

## Part 2: Borg Thresholds (Lines 10–19)

```python
BORG_ERROR_WARN  = 22.0   # p50 = baseline
BORG_ERROR_CRIT  = 30.0   # p75 = high risk
BORG_CPU_WARN    = 1.4    # p75
BORG_CPU_CRIT    = 4.1    # p90
```

**What's happening:**
These are "danger lines" calculated from Google's real data. For example:
- An error rate of **22%** is the normal baseline — half the time it's above this, half below.
- An error rate of **30%** means trouble — only 25% of the data is this bad or worse.
- These numbers were found by studying the actual Google dataset and calculating percentiles (p50 = median, p75 = top 25%, etc.).

---

## Part 3: Loading the Data (Lines 22–36)

```python
def _load_trace():
    try:
        df = pd.read_csv(TRACE_PATH)
        cols = ['cpu_util', 'mem_util', 'disk_io', 'net_latency',
                'error_rate', 'workload_trend', 'is_failure']
        df = df[cols].dropna().reset_index(drop=True)
        print(f"[rl_env] Loaded real Borg trace: {len(df):,} rows")
        return df
    except FileNotFoundError:
        print(f"[rl_env] WARNING: {TRACE_PATH} not found — using synthetic workload.")
        return None

_TRACE_DF = _load_trace()
```

**What's happening:**
- This function tries to open the cleaned data file (`processed_trace.csv`).
- It only keeps the 7 columns it actually needs.
- `dropna()` removes any rows with missing data (blank cells).
- If the file doesn't exist, it prints a warning and returns `None`, which tells the rest of the code to generate fake/synthetic data instead.
- `_TRACE_DF` is loaded **once** when the file is first imported, so it sits in memory ready to use — it doesn't re-read the CSV file every time.

---

## Part 4: The Environment Class — `__init__` (Lines 39–69)

```python
class CloudDisasterRecoveryEnv(gym.Env):
    def __init__(self, reward_config=None, max_steps=1000):
        super().__init__()

        # Actions: 0=NORMAL(30 steps), 1=WARNING(10), 2=CRITICAL(3)
        self.action_space = spaces.Discrete(3)
        self.intervals = {0: 30.0, 1: 10.0, 2: 3.0}

        # State: 8 features — normalised to [0,1] range
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32
        )

        self.max_steps = max_steps
        self.reward_config = reward_config or {}
        self._init_state()
```

**What's happening:**
- This creates the "game rules" for the AI.
- **3 actions (buttons the AI can press):**
  - `0 = NORMAL` → "Everything's fine, back up every 30 steps"
  - `1 = WARNING` → "Something looks off, back up every 10 steps"
  - `2 = CRITICAL` → "EMERGENCY! Back up every 3 steps"
- **8 observations (things the AI can see):** CPU usage, memory usage, disk activity, network delay, error rate, how long since last backup, SLA risk score, and workload trend. All squeezed into a 0-to-1 range so the AI's math doesn't get confused by wildly different scales.
- `max_steps = 1000` means each simulation episode lasts 1000 ticks.
- `reward_config` lets you tweak how much reward or penalty the AI gets (used for testing different strategies).

---

## Part 5: Resetting the Simulation — `_init_state` (Lines 72–113)

```python
def _init_state(self):
    self.current_step = 0
    self.cpu_util      = 1.4
    self.mem_util      = 1.9
    self.disk_io       = 25.0
    self.net_latency   = 4400.0
    self.error_rate    = 22.0
    self.backup_age    = 0.0
    self.sla_breach_risk = 0.22
    self.workload_trend  = 0.0

    self.cumulative_reward = 0.0
    self.rto = 2.0
    self.rpo = 0.0
    self.sla_breaches = 0
    self.total_steps_compliant = 0
    self.total_failures = 0

    # Trace playback: random start so each episode covers different data
    if _TRACE_DF is not None:
        self.trace_offset = random.randint(
            0, max(0, len(_TRACE_DF) - self.max_steps - 1)
        )
    else:
        self.trace_offset = 0

    # UI traces (rolling windows for dashboard charts)
    self.stationary_trace   = deque(maxlen=100)
    self.nonstationary_trace = deque(maxlen=100)
    self.backup_log = deque(maxlen=10)
```

**What's happening:**
- Every time a new episode starts, **everything gets reset to zero** — like restarting a game level.
- The initial values (cpu=1.4, error_rate=22.0, etc.) are set to the **real-world average** from Google's data, so the simulation starts in a realistic state.
- `trace_offset` picks a **random starting point** in the data file. This is important because if the AI always starts reading from row 1, it would just memorize that specific sequence. By randomizing the start, it learns general patterns instead of specific ones.
- The `deque` lists are rolling logs that keep only the most recent 100 entries — used to feed the live dashboard charts.

---

## Part 6: Helper Functions (Lines 116–137)

### Enabling a Workload Shift
```python
def enable_workload_shift(self, shift_at_step):
    self.workload_shift_enabled = True
    self.workload_shift_step = shift_at_step
```
This lets you schedule a sudden traffic spike at a specific step. For example: "At step 1000, suddenly double the server load." This tests whether the AI can adapt to unexpected changes.

### Getting the Next Data Row
```python
def _get_next_row(self):
    if _TRACE_DF is None:
        return None
    idx = (self.trace_offset + self.current_step) % len(_TRACE_DF)
    return _TRACE_DF.iloc[idx]
```
This reads the next line from the Google data file. The `% len(_TRACE_DF)` wraps around — if you reach the end of the data, it loops back to the beginning (like a playlist on repeat).

### Normalising the State
```python
def _normalise_state(self):
    return np.array([
        min(self.cpu_util   / 15.0, 1.0),    # CPU: divide by 15, cap at 1.0
        min(self.mem_util   / 15.0, 1.0),    # Memory: divide by 15, cap at 1.0
        min(self.disk_io    / 500.0, 1.0),   # Disk: divide by 500, cap at 1.0
        min(self.net_latency / 5000.0, 1.0), # Latency: divide by 5000, cap at 1.0
        min(self.error_rate / 100.0, 1.0),   # Errors: divide by 100, cap at 1.0
        min(self.backup_age / 30.0, 1.0),    # Backup age: divide by 30, cap at 1.0
        self.sla_breach_risk,                 # Already 0–1
        (self.workload_trend + 1.0) / 2.0    # Shift from (-1 to 1) → (0 to 1)
    ], dtype=np.float32)
```
**Why normalize?** Neural networks work best when all input numbers are in the same range (0 to 1). Without this, a net_latency of 4400 would completely overpower a cpu_util of 1.4, and the AI would ignore CPU entirely. By dividing each metric by its maximum expected value, they all become comparable.

---

## Part 7: The `step()` Function — The Core Game Loop (Lines 140–290)

This is the **most important function** in the entire project. Every time the clock ticks, this function runs. It has 8 phases:

### Phase 1: Stationary Reference Trace (Lines 144–147)
```python
self.stat_cpu = max(0, min(5, self.stat_cpu + random.uniform(-0.1, 0.1)))
self.stat_mem = max(0, min(5, self.stat_mem + random.uniform(-0.1, 0.1)))
self.stationary_trace.append({'cpu': self.stat_cpu, 'mem': self.stat_mem})
```
This generates a **calm, boring, predictable** CPU/memory trace as a comparison baseline. It just wiggles randomly by ±0.1 each step. Think of it as the "control group" in a science experiment — it shows what a perfectly stable server looks like so you can compare it against the chaotic real data.

### Phase 2: Load Real Server Data (Lines 149–199)
```python
self.backup_age += 1.0
row = self._get_next_row()

if row is not None:
    # REAL DATA
    self.cpu_util     = min(100.0, float(row['cpu_util']) * 1.86)
    self.mem_util     = min(100.0, float(row['mem_util']) * 4.47)
    self.disk_io      = float(row['disk_io'])
    self.net_latency  = float(row['net_latency'])
    self.error_rate   = float(row['error_rate'])
    self.workload_trend = float(row['workload_trend'])
    self.is_failure   = bool(row['is_failure'])

    # If workload shift is active, add extra stress
    if self.workload_shifted:
        self.cpu_util    = min(100, self.cpu_util + random.uniform(2, 10))
        self.error_rate  = min(100, self.error_rate + random.uniform(5, 15))
else:
    # SYNTHETIC FALLBACK (generates fake data if the CSV is missing)
    ...
```
**What's happening:**
- `backup_age += 1` — The "time since last backup" counter ticks up by 1 every step. This is critical: the longer you go without a backup, the more data you'd lose in a crash.
- If the real Google data file is available, it reads the next row and uses those values. The `* 1.86` and `* 4.47` scaling factors convert Google's raw numbers into a percentage scale that makes sense for our simulation.
- If a workload shift has been triggered, it **adds extra stress on top** of the real data (more CPU, more errors), simulating an unexpected traffic spike.
- If the data file is missing, the `else` block generates synthetic random data using Borg-calibrated averages so the simulation still works.

### Phase 3: Calculate SLA Breach Risk (Lines 201–208)
```python
self.sla_breach_risk = (
    (self.error_rate / 100.0) * 0.5 +       # 50% weight: error rate
    (min(self.backup_age, 30) / 30.0) * 0.3 + # 30% weight: backup age
    (self.cpu_util / 100.0) * 0.2            # 20% weight: CPU usage
)
self.sla_breach_risk = float(np.clip(self.sla_breach_risk, 0.0, 1.0))
```
**What's happening:**
This is a **composite danger score** from 0 (safe) to 1 (extremely dangerous). It's a weighted blend:
- **50%** comes from the error rate — the strongest predictor of a crash.
- **30%** comes from backup age — **this is the part the AI controls!** If the AI takes frequent backups, this stays low. If it ignores backups, this creeps up and signals danger.
- **20%** comes from CPU usage.

This is important because the AI can see this number. It learns: "When `sla_breach_risk` is climbing, I should take a backup soon."

### Phase 4: Backup Scheduler (Lines 215–228)
```python
interval = self.intervals[action]   # NORMAL=30, WARNING=10, CRITICAL=3
if self.backup_age >= interval:
    self.backup_triggered = True
    self.backup_log.append({
        'timestamp': self.current_step,
        'severity': ["NORMAL", "WARNING", "CRITICAL"][action],
        'rpo': self.backup_age
    })
    self.backup_age = 0.0    # Reset the timer!
    self.rpo = 0.0
```
**What's happening:**
- The AI chose an action (e.g., `CRITICAL` = action 2, which means "back up every 3 steps").
- The code checks: *Has enough time passed since the last backup?* (i.e., is `backup_age >= 3`?)
- If yes, a backup is triggered! The backup is logged, and the `backup_age` timer resets to 0.
- If the AI chose `NORMAL` (every 30 steps), it would take much longer before a backup happens, saving money but risking more data loss.

**In simple terms:** The AI is choosing how paranoid to be. CRITICAL = "back up constantly" (safe but expensive). NORMAL = "relax, back up rarely" (cheap but risky).

### Phase 5: RTO & RPO Calculation (Lines 229–231)
```python
self.rpo = self.backup_age
self.rto = 2.0 + 0.3 * self.rpo
```
**What's happening:**
- **RPO (Recovery Point Objective)** = How many minutes of data you'd lose if a crash happened *right now*. It equals the time since the last backup.
- **RTO (Recovery Time Objective)** = How long it would take to recover from a crash. The formula says it takes a base of 2 minutes plus 0.3 minutes for every minute of lost data. So if you haven't backed up in 10 steps, recovery takes 2 + 3 = 5 minutes.

### Phase 6: SLA Compliance Check (Lines 233–244)
```python
sla_met = (self.rto <= 5.0 and self.rpo <= 10.0)
if sla_met:
    self.total_steps_compliant += 1

if self.is_failure:
    self.total_failures += 1
    self.total_rto_sum += self.rto
    self.total_rpo_sum += self.rpo
    if not sla_met:
        self.sla_breaches += 1
```
**What's happening:**
- The **SLA (Service Level Agreement)** is a contract that says: "We promise recovery will take ≤ 5 minutes and we'll lose ≤ 10 minutes of data."
- Every step, the code checks: "Are we within the SLA right now?"
- If a crash happens AND the SLA is NOT met, it counts as a **breach** — the company broke its promise.
- These counters feed the final report card.

### Phase 7: Reward Calculation (Lines 246–262)
```python
if self.is_failure:
    if not sla_met:
        reward = rc.get('breach_penalty', -10.0)     # CRASHED + SLA BROKEN = -10
    else:
        reward = rc.get('survival_bonus', +1.0)       # CRASHED but SLA met = +1
elif self.backup_triggered:
    reward = rc.get('backup_cost', -0.1)              # Backup taken (costs money) = -0.1
else:
    reward = rc.get('idle_bonus', +0.01)              # Nothing happened = +0.01
```
**What's happening:**
This is **how the AI learns right from wrong**. It's like giving a dog a treat or a scolding:
- **Server crashed AND backup was too old** → `-10 points` (very bad! The AI failed its job)
- **Server crashed BUT backup was recent** → `+1 point` (good! The AI predicted the crash and prepared)
- **No crash, but a backup was taken** → `-0.1 points` (small cost — backups aren't free, so don't overdo it)
- **No crash, no backup** → `+0.01 points` (tiny reward for staying calm when nothing is wrong)

The key insight: the AI must learn the **balance** between -0.1 (cost of backing up) and -10.0 (cost of NOT backing up when a crash comes). Over thousands of episodes, it figures out the sweet spot.

### Phase 8: Logging & Return (Lines 264–290)
```python
self.full_action_log.append({
    'step': self.current_step,
    'action': action,
    'is_failure': self.is_failure,
    'backup_triggered': self.backup_triggered,
    'rto': self.rto, 'rpo': self.rpo,
    'sla_met': sla_met
})

done = self.current_step >= self.max_steps
state = self._normalise_state()
info = {'is_failure': self.is_failure, 'backup_triggered': self.backup_triggered}
return state, reward, done, False, info
```
**What's happening:**
- Every step's details are saved to a log (used later to draw graphs).
- The function returns 5 things back to whoever called it:
  1. `state` — the new normalised server health numbers (what the AI sees next)
  2. `reward` — the score for this step (+1, -10, etc.)
  3. `done` — True if the episode is over (reached max steps)
  4. `False` — "truncated" flag (always False here, it's a Gymnasium convention)
  5. `info` — extra details like whether a failure or backup happened

---

## Part 8: `get_metrics()` and `reset()` (Lines 293–307)

```python
def get_metrics(self):
    total = max(self.current_step, 1)
    return {
        'sla_compliance_rate': (self.total_steps_compliant / total) * 100.0,
        'mean_rto_min': self.total_rto_sum / max(self.total_failures, 1),
        'mean_rpo_min': self.total_rpo_sum / max(self.total_failures, 1),
        'total_sla_breaches': self.sla_breaches,
        'total_failures': self.total_failures,
        'cumulative_reward': self.cumulative_reward
    }

def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self._init_state()
    return self._normalise_state(), {}
```

**What's happening:**
- `get_metrics()` calculates the **final report card** after an episode ends:
  - *What percentage of the time were we within SLA?*
  - *On average, how fast did we recover (RTO)?*
  - *On average, how much data did we lose (RPO)?*
  - *How many times did we break our promise (SLA breaches)?*
- `reset()` wipes everything clean and starts a new episode from scratch. This is called at the beginning of every training episode and every evaluation run.

---

## Summary: The Full Cycle in One Step

```
1. Clock ticks → backup_age goes up by 1
2. Read the next line of Google server data (CPU, errors, etc.)
3. Calculate the danger score (sla_breach_risk)
4. Check: did the AI's chosen action trigger a backup? If so, reset backup_age to 0
5. Calculate RTO and RPO
6. Check: are we within SLA right now?
7. Did a crash happen?
   → YES + SLA broken = -10 penalty
   → YES + SLA met    = +1 reward
   → NO  + backup     = -0.1 cost
   → NO  + idle       = +0.01 bonus
8. Log everything, return the new state to the AI
```

The AI repeats this loop 1,000 times per episode and plays hundreds of episodes during training, gradually learning when to press the CRITICAL button and when to stay calm.
