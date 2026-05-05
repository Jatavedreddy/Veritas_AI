"""
Veritas — Synthetic Institutional Transaction Data Generator
═══════════════════════════════════════════════════════════════
Generates realistic simulated financial transaction data with
injected anomalous patterns for ML anomaly-detection training.

Anomaly Scenarios Injected:
  1. Structuring (Smurfing)   – Clusters of transactions just below $10k
  2. Offshore Midnight Sweep  – Massive overnight wires to tax-haven regions
  3. Velocity Spike (Burst)   – Rapid-fire micro-transactions from one account
  4. Round-Trip Wash           – Funds looped between two accounts
  5. Geographic Mismatch       – Rare high-value SWIFT to sanctioned regions

Usage:
    python -m data.synthetic_data_gen          # from project root
    python data/synthetic_data_gen.py          # direct execution

Output:
    data/raw/transactions.csv
    data/raw/transactions.json
"""

import os
import uuid
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ─── Configuration ─────────────────────────────────────────────────────────────

SEED = 42
NUM_NORMAL_ROWS = 5000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF"]
TRANSACTION_TYPES = ["ACH", "Wire", "SWIFT"]

NORMAL_REGIONS = [
    "North America", "Western Europe", "East Asia",
    "South America", "Southeast Asia", "Oceania",
]
OFFSHORE_REGIONS = [
    "Cayman Islands", "British Virgin Islands",
    "Panama", "Isle of Man", "Bermuda",
]
SANCTIONED_REGIONS = [
    "North Korea", "Iran", "Syria", "Crimea",
]
ALL_REGIONS = NORMAL_REGIONS + OFFSHORE_REGIONS + SANCTIONED_REGIONS

# ─── Helpers ───────────────────────────────────────────────────────────────────

fake = Faker()
Faker.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def _account_id() -> str:
    """Generate a realistic-looking institutional account ID."""
    prefix = random.choice(["INST", "CORP", "FND", "HNW", "PB"])
    return f"{prefix}-{fake.bothify(text='####-####-####')}"


def _txn_id() -> str:
    """UUID v4 transaction identifier."""
    return str(uuid.uuid4())


def _random_timestamp(start: datetime, end: datetime) -> datetime:
    """Random datetime between start and end."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


# Pre-generate a pool of accounts for reuse (institutional = fewer accounts, many txns)
ACCOUNT_POOL = [_account_id() for _ in range(200)]


# ─── Normal Transaction Generator ─────────────────────────────────────────────

def generate_normal_transactions(n: int) -> pd.DataFrame:
    """Generate n normal, benign institutional transactions."""

    print(f"  ▸ Generating {n:,} normal transactions …")

    # Date range: past 180 days
    end_date = datetime(2026, 5, 1)
    start_date = end_date - timedelta(days=180)

    records = []
    for _ in range(n):
        from_acct = random.choice(ACCOUNT_POOL)
        to_acct = random.choice([a for a in ACCOUNT_POOL if a != from_acct])

        # Log-normal distribution for realistic amounts ($50 – $500k range)
        amount = round(np.random.lognormal(mean=8.5, sigma=1.8), 2)
        amount = min(amount, 5_000_000)  # cap at $5M
        amount = max(amount, 10.0)       # floor at $10

        records.append({
            "transaction_id": _txn_id(),
            "timestamp": _random_timestamp(start_date, end_date),
            "from_account": from_acct,
            "to_account": to_acct,
            "amount": amount,
            "currency": random.choices(CURRENCIES, weights=[50, 20, 15, 10, 5])[0],
            "transaction_type": random.choices(TRANSACTION_TYPES, weights=[50, 35, 15])[0],
            "region": random.choices(NORMAL_REGIONS, weights=[35, 25, 20, 10, 5, 5])[0],
            "is_synthetic_anomaly": False,
            "anomaly_type": "none",
        })

    return pd.DataFrame(records)


# ─── Anomaly Injectors ────────────────────────────────────────────────────────

def inject_structuring(n: int = 120) -> pd.DataFrame:
    """
    Scenario 1 — Structuring (Smurfing)
    Multiple transactions of exactly $9,999 or just below $10,000
    from the same account within a short window, designed to evade
    CTR (Currency Transaction Report) thresholds.
    """
    print(f"  ▸ Injecting {n} structuring anomalies …")

    # Pick a handful of "dirty" accounts
    dirty_accounts = random.sample(ACCOUNT_POOL, 6)
    base_date = datetime(2026, 3, 15)

    records = []
    for acct in dirty_accounts:
        burst_size = n // len(dirty_accounts)
        window_start = base_date + timedelta(days=random.randint(0, 30))

        for i in range(burst_size):
            # Amount clusters: $9,990 – $9,999
            amount = round(random.uniform(9_990.00, 9_999.99), 2)
            ts = window_start + timedelta(minutes=random.randint(5, 120) * (i + 1))

            records.append({
                "transaction_id": _txn_id(),
                "timestamp": ts,
                "from_account": acct,
                "to_account": random.choice([a for a in ACCOUNT_POOL if a != acct]),
                "amount": amount,
                "currency": "USD",
                "transaction_type": "ACH",
                "region": "North America",
                "is_synthetic_anomaly": True,
                "anomaly_type": "structuring",
            })

    return pd.DataFrame(records)


def inject_offshore_midnight_sweep(n: int = 80) -> pd.DataFrame:
    """
    Scenario 2 — Offshore Midnight Sweep
    Massive wire transfers ($200k–$2M) executed between midnight–4 AM
    to offshore tax-haven regions.
    """
    print(f"  ▸ Injecting {n} offshore midnight sweep anomalies …")

    records = []
    base_date = datetime(2026, 2, 1)

    for _ in range(n):
        day_offset = random.randint(0, 90)
        hour = random.randint(0, 3)  # midnight – 3 AM
        minute = random.randint(0, 59)
        ts = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)

        records.append({
            "transaction_id": _txn_id(),
            "timestamp": ts,
            "from_account": random.choice(ACCOUNT_POOL),
            "to_account": _account_id(),  # new unknown external account
            "amount": round(random.uniform(200_000, 2_000_000), 2),
            "currency": random.choice(["USD", "CHF", "EUR"]),
            "transaction_type": "Wire",
            "region": random.choice(OFFSHORE_REGIONS),
            "is_synthetic_anomaly": True,
            "anomaly_type": "offshore_midnight_sweep",
        })

    return pd.DataFrame(records)


def inject_velocity_spike(n: int = 100) -> pd.DataFrame:
    """
    Scenario 3 — Velocity Spike (Burst)
    A single account fires 50+ micro-transactions ($1–$50) within minutes,
    suggesting bot-driven layering or DDoS-style settlement attacks.
    """
    print(f"  ▸ Injecting {n} velocity spike anomalies …")

    # Two accounts exhibit this behaviour
    spike_accounts = random.sample(ACCOUNT_POOL, 2)

    records = []
    for acct in spike_accounts:
        burst_size = n // 2
        burst_start = datetime(2026, 4, 10, 14, 30)  # during trading hours

        for i in range(burst_size):
            ts = burst_start + timedelta(seconds=random.randint(1, 5) * (i + 1))
            records.append({
                "transaction_id": _txn_id(),
                "timestamp": ts,
                "from_account": acct,
                "to_account": random.choice([a for a in ACCOUNT_POOL if a != acct]),
                "amount": round(random.uniform(1.00, 50.00), 2),
                "currency": "USD",
                "transaction_type": "ACH",
                "region": "North America",
                "is_synthetic_anomaly": True,
                "anomaly_type": "velocity_spike",
            })

    return pd.DataFrame(records)


def inject_round_trip_wash(n: int = 60) -> pd.DataFrame:
    """
    Scenario 4 — Round-Trip Wash Trading
    Funds are transferred A→B then B→A with near-identical amounts
    within a short window, suggesting wash trading or money laundering.
    """
    print(f"  ▸ Injecting {n} round-trip wash anomalies …")

    acct_a, acct_b = random.sample(ACCOUNT_POOL, 2)
    base_date = datetime(2026, 3, 1)

    records = []
    for i in range(n // 2):
        amount = round(random.uniform(50_000, 500_000), 2)
        ts1 = base_date + timedelta(days=random.randint(0, 60), hours=random.randint(9, 16))
        ts2 = ts1 + timedelta(minutes=random.randint(2, 30))

        # Leg 1: A → B
        records.append({
            "transaction_id": _txn_id(),
            "timestamp": ts1,
            "from_account": acct_a,
            "to_account": acct_b,
            "amount": amount,
            "currency": "USD",
            "transaction_type": "Wire",
            "region": "North America",
            "is_synthetic_anomaly": True,
            "anomaly_type": "round_trip_wash",
        })

        # Leg 2: B → A (slightly different amount to look less obvious)
        records.append({
            "transaction_id": _txn_id(),
            "timestamp": ts2,
            "from_account": acct_b,
            "to_account": acct_a,
            "amount": round(amount * random.uniform(0.995, 1.005), 2),
            "currency": "USD",
            "transaction_type": "Wire",
            "region": "North America",
            "is_synthetic_anomaly": True,
            "anomaly_type": "round_trip_wash",
        })

    return pd.DataFrame(records)


def inject_geographic_mismatch(n: int = 40) -> pd.DataFrame:
    """
    Scenario 5 — Geographic Mismatch / Sanctions Violation
    High-value SWIFT transfers to sanctioned or unusual regions,
    flagged for OFAC / sanctions screening.
    """
    print(f"  ▸ Injecting {n} geographic mismatch anomalies …")

    records = []
    base_date = datetime(2026, 1, 15)

    for _ in range(n):
        ts = base_date + timedelta(days=random.randint(0, 120), hours=random.randint(0, 23))
        records.append({
            "transaction_id": _txn_id(),
            "timestamp": ts,
            "from_account": random.choice(ACCOUNT_POOL),
            "to_account": _account_id(),
            "amount": round(random.uniform(100_000, 3_000_000), 2),
            "currency": random.choice(["EUR", "CHF", "GBP"]),
            "transaction_type": "SWIFT",
            "region": random.choice(SANCTIONED_REGIONS),
            "is_synthetic_anomaly": True,
            "anomaly_type": "geographic_mismatch",
        })

    return pd.DataFrame(records)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def generate_dataset() -> pd.DataFrame:
    """Assemble the full synthetic dataset with injected anomalies."""

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Veritas — Synthetic Transaction Data Generator        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 1. Normal transactions
    df_normal = generate_normal_transactions(NUM_NORMAL_ROWS)

    # 2. Inject anomaly patterns
    df_structuring = inject_structuring(120)
    df_offshore    = inject_offshore_midnight_sweep(80)
    df_velocity    = inject_velocity_spike(100)
    df_wash        = inject_round_trip_wash(60)
    df_geo         = inject_geographic_mismatch(40)

    # 3. Combine & shuffle
    df = pd.concat(
        [df_normal, df_structuring, df_offshore, df_velocity, df_wash, df_geo],
        ignore_index=True,
    )
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 4. Sort by timestamp for realism
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame) -> None:
    """Persist dataset to CSV and JSON."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "transactions.csv")
    json_path = os.path.join(OUTPUT_DIR, "transactions.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, date_format="iso")

    print(f"\n  ✔ CSV  saved → {csv_path}")
    print(f"  ✔ JSON saved → {json_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the generated dataset."""

    total = len(df)
    anomalies = df["is_synthetic_anomaly"].sum()
    normal = total - anomalies

    print("\n─── Dataset Summary ───────────────────────────────────────")
    print(f"  Total transactions : {total:,}")
    print(f"  Normal             : {normal:,}  ({normal/total:.1%})")
    print(f"  Anomalous          : {anomalies:,}  ({anomalies/total:.1%})")
    print()
    print("  Anomaly Breakdown:")
    for atype, count in df[df["is_synthetic_anomaly"]]["anomaly_type"].value_counts().items():
        print(f"    • {atype:<30s} {count:>5,}")
    print()
    print("  Amount Statistics:")
    print(f"    Min    : ${df['amount'].min():>14,.2f}")
    print(f"    Median : ${df['amount'].median():>14,.2f}")
    print(f"    Mean   : ${df['amount'].mean():>14,.2f}")
    print(f"    Max    : ${df['amount'].max():>14,.2f}")
    print()
    print("  Sample Rows:")
    print(df[["transaction_id", "timestamp", "amount", "transaction_type",
              "region", "is_synthetic_anomaly", "anomaly_type"]].head(10).to_string(index=False))
    print("───────────────────────────────────────────────────────────\n")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_dataset()
    print_summary(df)
    save_dataset(df)
    print("  🚀 Data generation complete. Ready for Phase 2 (ML).\n")
