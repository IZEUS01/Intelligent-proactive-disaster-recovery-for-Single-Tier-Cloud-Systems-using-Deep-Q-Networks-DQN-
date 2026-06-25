# pyrefly: ignore [missing-import]
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd
from collections import deque

TRACE_PATH = "data/processed_trace.csv"

# ── Borg-calibrated thresholds (derived from percentile analysis) ─────────────
# error_rate: p25=15, p50=20, p75=30, failure_mean=26.7, non_fail_mean=21.7
# cpu_scaled: p50=0.2, p75=1.4, p99=14.8  (very skewed, low signal)
# net_latency: 75th pct = 5000 (maxed out, not useful)
# Key insight: error_rate is the PRIMARY signal. backup_age + sla_breach_risk
# are the AGENT-CONTROLLED signals the DQN must learn to exploit.
BORG_ERROR_WARN  = 22.0   # p50 = baseline
BORG_ERROR_CRIT  = 30.0   # p75 = high risk
BORG_CPU_WARN    = 1.4    # p75
BORG_CPU_CRIT    = 4.1    # p90


def _load_trace():
    """Load preprocessed Borg trace. Returns DataFrame or None."""
    try:
        df = pd.read_csv(TRACE_PATH)
        cols = ['cpu_util', 'mem_util', 'disk_io', 'net_latency',
                'error_rate', 'workload_trend', 'is_failure']
        df = df[cols].dropna().reset_index(drop=True)
        print(f"[rl_env] Loaded real Borg trace: {len(df):,} rows")
        return df
    except FileNotFoundError:
        print(f"[rl_env] WARNING: {TRACE_PATH} not found — using synthetic workload.")
        print(f"[rl_env]   Run: python preprocess.py")
        return None

_TRACE_DF = _load_trace()


class CloudDisasterRecoveryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_config=None, max_steps=1000):
        super().__init__()

        # Actions: 0=NORMAL(30 steps), 1=WARNING(10), 2=CRITICAL(3)
        # CRITICAL interval set to 3 (not 2) to make DQN earn its advantage
        self.action_space = spaces.Discrete(3)
        self.intervals = {0: 30.0, 1: 10.0, 2: 3.0}

        # State: 8 features — normalised to [0,1] range for stable DQN learning
        # [cpu_util/15, mem_util/15, disk_io/500, net_latency/5000,
        #  error_rate/100, backup_age/30, sla_breach_risk, workload_trend+1/2]
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32
        )

        self.max_steps = max_steps

        # Reward config (used for ablation Graph 7 only)
        self.reward_config = reward_config or {}

        # Workload shift injection (Graph 4)
        self.workload_shift_enabled = False
        self.workload_shift_step = None
        self.workload_shifted = False

        self._init_state()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_state(self):
        self.current_step = 0
        # Raw (unnormalised) telemetry — updated each step from trace or synth
        self.cpu_util      = 1.4    # Borg p75
        self.mem_util      = 1.9
        self.disk_io       = 25.0
        self.net_latency   = 4400.0
        self.error_rate    = 22.0   # Borg baseline
        self.backup_age    = 0.0    # controlled by agent
        self.sla_breach_risk = 0.22
        self.workload_trend  = 0.0

        self.cumulative_reward = 0.0
        self.rto = 2.0
        self.rpo = 0.0
        self.sla_breaches = 0
        self.total_steps_compliant = 0
        self.total_failures = 0
        self.total_rto_sum = 0.0
        self.total_rpo_sum = 0.0
        self.is_failure = False
        self.backup_triggered = False
        self.workload_shifted = False

        # Trace playback: random start so each episode covers different data
        if _TRACE_DF is not None:
            self.trace_offset = random.randint(
                0, max(0, len(_TRACE_DF) - self.max_steps - 1)
            )
        else:
            self.trace_offset = 0

        # UI traces
        self.stationary_trace   = deque(maxlen=100)
        self.nonstationary_trace = deque(maxlen=100)
        self.backup_log = deque(maxlen=10)
        self.stat_cpu = 1.4
        self.stat_mem = 1.9

        # Full logs (Graphs 5, 8)
        self.full_action_log = []
        self.full_state_log  = []

    # ─────────────────────────────────────────────────────────────────────────
    def enable_workload_shift(self, shift_at_step):
        self.workload_shift_enabled = True
        self.workload_shift_step = shift_at_step

    def _get_next_row(self):
        if _TRACE_DF is None:
            return None
        idx = (self.trace_offset + self.current_step) % len(_TRACE_DF)
        return _TRACE_DF.iloc[idx]

    def _normalise_state(self):
        """Normalise raw telemetry to [0,1] for stable neural network input."""
        return np.array([
            min(self.cpu_util   / 15.0, 1.0),   # p99=14.8 → cap at 15
            min(self.mem_util   / 15.0, 1.0),
            min(self.disk_io    / 500.0, 1.0),
            min(self.net_latency / 5000.0, 1.0),
            min(self.error_rate / 100.0, 1.0),
            min(self.backup_age / 30.0, 1.0),   # 30 = NORMAL interval
            self.sla_breach_risk,                 # already 0–1
            (self.workload_trend + 1.0) / 2.0    # shift -1..1 → 0..1
        ], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        self.is_failure = False
        self.backup_triggered = False

        # 1. Stationary reference trace (always synthetic for comparison)
        self.stat_cpu = max(0, min(5, self.stat_cpu + random.uniform(-0.1, 0.1)))
        self.stat_mem = max(0, min(5, self.stat_mem + random.uniform(-0.1, 0.1)))
        self.stationary_trace.append({'cpu': self.stat_cpu, 'mem': self.stat_mem})

        # 2. Load real Borg telemetry or fall back to synthetic
        self.backup_age += 1.0
        row = self._get_next_row()

        if row is not None:
            # ── REAL DATA: raw Borg values, scaled to usable range ──────────
            # CPU: Borg raw 0–54% → scale so p99(14.8%) stays meaningful
            self.cpu_util     = min(100.0, float(row['cpu_util']) * 1.86)
            self.mem_util     = min(100.0, float(row['mem_util']) * 4.47)
            self.disk_io      = float(row['disk_io'])
            self.net_latency  = float(row['net_latency'])
            self.error_rate   = float(row['error_rate'])
            self.workload_trend = float(row['workload_trend'])
            self.is_failure   = bool(row['is_failure'])

            # Workload shift: add synthetic load on top of real data
            if (self.workload_shift_enabled and
                    self.workload_shift_step is not None and
                    self.current_step >= self.workload_shift_step and
                    not self.workload_shifted):
                self.workload_shifted = True
            if self.workload_shifted:
                self.cpu_util    = min(100, self.cpu_util + random.uniform(2, 10))
                self.error_rate  = min(100, self.error_rate + random.uniform(5, 15))

        else:
            # ── SYNTHETIC FALLBACK ───────────────────────────────────────────
            self.workload_trend += random.uniform(-0.05, 0.05)
            self.workload_trend  = max(-1.0, min(1.0, self.workload_trend))

            if (self.workload_shift_enabled and
                    self.workload_shift_step is not None and
                    self.current_step >= self.workload_shift_step and
                    not self.workload_shifted):
                self.workload_shifted = True
                self.workload_trend = 0.6

            # Borg-calibrated base values
            base_error = 22.0 + self.workload_trend * 5
            self.error_rate  = max(0, min(100, base_error + random.gauss(0, 4)))
            self.cpu_util    = max(0, 1.4 + self.workload_trend * 2 + random.gauss(0, 1))
            self.mem_util    = max(0, 1.9 + self.workload_trend * 1.5 + random.gauss(0, 1))
            self.disk_io     = max(0, 25.0 + self.workload_trend * 10 + random.gauss(0, 5))
            self.net_latency = max(1000, min(5000, 4400 + random.gauss(0, 300)))

            # Failure probability calibrated to Borg's 22.8% failure rate
            # Error_rate is the main predictor: higher error → more failures
            fail_prob = 0.10 + 0.005 * max(0, self.error_rate - 22.0)
            if self.workload_shifted:
                fail_prob += 0.05
            self.is_failure = random.random() < fail_prob

        # 3. Compute SLA breach risk (agent-controlled: backup_age matters!)
        #    DQN sees this signal and learns to keep backup_age low
        self.sla_breach_risk = (
            (self.error_rate / 100.0) * 0.5 +       # strongest Borg signal
            (min(self.backup_age, 30) / 30.0) * 0.3 + # agent-controlled!
            (self.cpu_util / 100.0) * 0.2
        )
        self.sla_breach_risk = float(np.clip(self.sla_breach_risk, 0.0, 1.0))

        self.nonstationary_trace.append({
            'cpu': self.cpu_util, 'mem': self.mem_util,
            'failure': self.is_failure, 'backup': False
        })

        # 4. Backup scheduler (agent controls via action)
        interval = self.intervals[action]
        if self.backup_age >= interval:
            self.backup_triggered = True
            self.backup_log.append({
                'timestamp': self.current_step,
                'severity': ["NORMAL", "WARNING", "CRITICAL"][action],
                'rpo': self.backup_age
            })
            self.backup_age = 0.0
            self.rpo = 0.0
            if self.nonstationary_trace:
                self.nonstationary_trace[-1]['backup'] = True

        # 5. RTO / RPO
        self.rpo = self.backup_age
        self.rto = 2.0 + 0.3 * self.rpo   # RTO grows with RPO

        # 6. SLA compliance: RTO ≤ 5min AND RPO ≤ 10min
        #    (tighter than before: 10 instead of 15 — forces DQN to be proactive)
        sla_met = (self.rto <= 5.0 and self.rpo <= 10.0)
        if sla_met:
            self.total_steps_compliant += 1

        if self.is_failure:
            self.total_failures  += 1
            self.total_rto_sum   += self.rto
            self.total_rpo_sum   += self.rpo
            if not sla_met:
                self.sla_breaches += 1

        # 7. Reward  R(t) — calibrated for Borg's 22.8% failure rate
        #    R(t) = { -10   SLA breached at failure
        #           {  +1   failure but SLA met (DQN predicted and backed up)
        #           { -0.1  backup triggered (cost)
        #           { +0.01 idle (no failure, no backup)
        rc = self.reward_config
        if self.is_failure:
            if not sla_met:
                reward = rc.get('breach_penalty', -10.0)
            else:
                reward = rc.get('survival_bonus', +1.0)
        elif self.backup_triggered:
            reward = rc.get('backup_cost', -0.1)
        else:
            reward = rc.get('idle_bonus', +0.01)

        self.cumulative_reward += reward
        self.current_step += 1

        # 8. Logging
        self.full_action_log.append({
            'step': self.current_step,
            'action': action,
            'is_failure': self.is_failure,
            'backup_triggered': self.backup_triggered,
            'rto': self.rto,
            'rpo': self.rpo,
            'sla_met': sla_met
        })
        self.full_state_log.append({
            'cpu_util': self.cpu_util,
            'mem_util': self.mem_util,
            'disk_io': self.disk_io,
            'net_latency': self.net_latency,
            'error_rate': self.error_rate,
            'backup_age': self.backup_age,
            'sla_breach_risk': self.sla_breach_risk,
            'workload_trend': self.workload_trend,
            'sla_breach': not sla_met
        })

        done = self.current_step >= self.max_steps
        state = self._normalise_state()
        info  = {'is_failure': self.is_failure, 'backup_triggered': self.backup_triggered}
        return state, reward, done, False, info

    # ─────────────────────────────────────────────────────────────────────────
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
