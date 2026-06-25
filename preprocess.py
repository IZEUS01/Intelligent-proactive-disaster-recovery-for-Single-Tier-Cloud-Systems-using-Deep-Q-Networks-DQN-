"""
preprocess.py — Preprocess the Google Borg Cluster Trace dataset
into the 8-feature format used by CloudDisasterRecoveryEnv.

Input:  data/borg_traces_data.csv
Output: data/processed_trace.csv

Feature mapping:
  csv column                    → env feature
  ─────────────────────────────────────────────────
  average_usage['cpus']  ×100  → cpu_util    (%)
  average_usage['memory'] ×100 → mem_util    (%)
  maximum_usage['cpus']  ×1000 → disk_io     (MB/s proxy)
  end_time - start_time (ms→s) → net_latency (ms)
  failed (0/1 rolling mean)    → error_rate  (%)
  [computed per row]           → backup_age  (steps since last backup)
  [computed per row]           → sla_breach_risk (composite)
  cpu_util rolling slope       → workload_trend  (-1 to 1)
  event == FAIL                → is_failure  (bool)
"""

import pandas as pd
import numpy as np
import ast
import os

RAW_PATH  = "data/borg_traces_data.csv"
OUT_PATH  = "data/processed_trace.csv"
CHUNK_SIZE = 50_000   # process in chunks — file is 328 MB

def parse_dict_field(series, key):
    """Extract a numeric key from a column of stringified dicts."""
    def _get(s):
        try:
            d = ast.literal_eval(str(s))
            v = d.get(key, np.nan)
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan
    return series.apply(_get)


def process_chunk(df):
    out = pd.DataFrame()

    # ── cpu_util (%): average CPU usage × 100, clamp 0–100
    cpu_raw = parse_dict_field(df['average_usage'], 'cpus')
    out['cpu_util'] = (cpu_raw * 100).clip(0, 100).fillna(0)

    # ── mem_util (%): average memory usage × 100, clamp 0–100
    mem_raw = parse_dict_field(df['average_usage'], 'memory')
    out['mem_util'] = (mem_raw * 100).clip(0, 100).fillna(0)

    # ── disk_io (MB/s proxy): max CPU × 1000 as a throughput proxy
    #    (Borg doesn't log disk separately; max_cpu correlates with I/O burst)
    cpu_max = parse_dict_field(df['maximum_usage'], 'cpus')
    out['disk_io'] = (cpu_max * 1000).clip(0, 5000).fillna(0)

    # ── net_latency (ms): task duration in ms (end_time – start_time are in µs)
    #    Borg timestamps are in microseconds; divide by 1000 to get ms, cap at 5000
    duration_ms = (df['end_time'] - df['start_time']).abs() / 1000.0
    out['net_latency'] = duration_ms.clip(1, 5000).fillna(20)

    # ── error_rate (%): rolling mean of 'failed' flag × 100, window=20
    out['error_rate'] = (
        df['failed'].fillna(0).rolling(window=20, min_periods=1).mean() * 100
    ).clip(0, 100)

    # ── backup_age (min): steps since a failure event — resets to 0 at each FAIL
    is_fail = (df['event'] == 'FAIL').fillna(False).astype(int)
    backup_age = []
    age = 0
    for f in is_fail:
        if f:
            age = 0
        else:
            age += 1
        backup_age.append(min(age, 10000))
    out['backup_age'] = backup_age

    # ── sla_breach_risk: composite from cpu, error, latency
    out['sla_breach_risk'] = (
        (out['cpu_util'] / 100.0) * 0.4 +
        (out['error_rate'] / 100.0) * 0.4 +
        (out['net_latency'] / 5000.0) * 0.2
    ).clip(0, 1)

    # ── workload_trend: normalised rolling slope of cpu_util, window=10
    cpu_roll = out['cpu_util'].rolling(window=10, min_periods=2)
    slope = cpu_roll.apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0,
        raw=True
    )
    # Normalise to [-1, 1] using a ±5 %/step range
    out['workload_trend'] = (slope / 5.0).clip(-1, 1).fillna(0)

    # ── is_failure: ground-truth failure label for the environment
    out['is_failure'] = is_fail

    # ── time reference (for ordering)
    out['time'] = df['time'].values

    return out.reset_index(drop=True)


def main():
    os.makedirs("data", exist_ok=True)

    print(f"Reading {RAW_PATH}  …")
    total_rows = sum(1 for _ in open(RAW_PATH)) - 1  # minus header
    print(f"Total rows in CSV: {total_rows:,}")

    chunks = []
    reader = pd.read_csv(
        RAW_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    processed = 0
    for i, chunk in enumerate(reader):
        proc = process_chunk(chunk)
        chunks.append(proc)
        processed += len(chunk)
        pct = processed / total_rows * 100
        print(f"  Processed {processed:>8,} / {total_rows:,} rows  ({pct:.1f}%)")

    df_out = pd.concat(chunks, ignore_index=True)

    # Sort by time so playback is chronological
    df_out.sort_values('time', inplace=True)
    df_out.drop(columns=['time'], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df_out):,} rows → {OUT_PATH}")
    print("\nColumn stats:")
    print(df_out.describe().to_string())


if __name__ == "__main__":
    main()
