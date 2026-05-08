import random
import sys
import time
import uuid
from datetime import datetime, timezone

import requests


# BASE_URL = "http://127.0.0.1:8080/api/v1"
BASE_URL = "https://veritas-platform-web-app-fhg4ekdkcgbdfpbc.koreacentral-01.azurewebsites.net/api/v1"
INGEST_URL = f"{BASE_URL}/ingest"
PREDICT_URL = f"{BASE_URL}/predict"
SLEEP_SECONDS = 1.5
ANOMALY_RATE = 0.05


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"


NORMAL_REGIONS = [
    "North America",
    "Europe",
    "Asia Pacific",
    "Middle East",
    "Latin America",
]
ANOMALY_REGIONS = [
    "High-Risk Jurisdiction",
    "Sanctioned Corridor",
    "Watchlist Zone",
]
NORMAL_CURRENCIES = ["USD", "EUR", "GBP", "CAD", "SGD", "JPY", "AED"]
ANOMALY_CURRENCIES = ["USD", "EUR", "CHF"]
NORMAL_TYPES = ["ACH", "Wire", "Credit Card", "SEPA", "Payroll", "POS"]
ANOMALY_TYPES = ["SWIFT", "Offshore Wire", "Private Banking Transfer"]
NORMAL_HOLDERS = [
    "Northstar Retail LLC",
    "Apex MedSupply",
    "Blue Harbor Logistics",
    "Kingsley Advisory",
    "Sunline Manufacturing",
    "Harborview Foods",
]
ANOMALY_HOLDERS = [
    "Shell Company XYZ",
    "Dormant Holdings SPC",
    "Black Reef Trust",
]


def banner():
    print(f"{Colors.MAGENTA}{Colors.BOLD}VERITAS LIVE SIMULATOR{Colors.RESET}")
    print(
        f"{Colors.DIM}Streaming transactions to {BASE_URL} every "
        f"{SLEEP_SECONDS:.1f}s{Colors.RESET}"
    )


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def choose_normal_amount():
    return round(random.uniform(100, 3000), 2)


def choose_anomaly_amount():
    return round(
        random.choice([9999999, 7500000, 12500000, 4200000]) + random.uniform(0, 999),
        2,
    )


def build_transaction():
    is_anomaly = random.random() < ANOMALY_RATE
    transaction_id = f"txn-live-{uuid.uuid4().hex[:12]}"

    if is_anomaly:
        amount = choose_anomaly_amount()
        currency = random.choice(ANOMALY_CURRENCIES)
        transaction_type = random.choice(ANOMALY_TYPES)
        region = random.choice(ANOMALY_REGIONS)
        holder = random.choice(ANOMALY_HOLDERS)
        historical_avg = round(random.uniform(40, 250), 2)
    else:
        amount = choose_normal_amount()
        currency = random.choice(NORMAL_CURRENCIES)
        transaction_type = random.choice(NORMAL_TYPES)
        region = random.choice(NORMAL_REGIONS)
        holder = random.choice(NORMAL_HOLDERS)
        historical_avg = round(amount * random.uniform(0.75, 1.15), 2)

    return {
        "transaction_id": transaction_id,
        "timestamp": iso_now(),
        "amount": amount,
        "currency": currency,
        "transaction_type": transaction_type,
        "region": region,
        "ingest_source": "live_simulator",
        "account_metadata": {
            "account_holder": holder,
            "historical_avg_transaction": historical_avg,
        },
    }, is_anomaly


def status_line(transaction, is_anomaly, prediction_result):
    marker = "ANOMALY" if is_anomaly else "NORMAL "
    color = Colors.RED if is_anomaly else Colors.GREEN
    predicted = "flagged" if prediction_result.get("is_anomaly") else "clear"

    return (
        f"{color}{Colors.BOLD}[{marker}]{Colors.RESET} "
        f"{Colors.CYAN}{transaction['transaction_id']}{Colors.RESET} "
        f"{transaction['currency']} {transaction['amount']:,.2f} "
        f"{Colors.DIM}{transaction['transaction_type']} | {transaction['region']}{Colors.RESET} "
        f"{Colors.BLUE}model:{predicted}{Colors.RESET}"
    )


def post_json(url, payload, timeout=10):
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main():
    banner()

    while True:
        try:
            transaction, intended_anomaly = build_transaction()
            payload = {"transactions": [transaction]}

            ingest_result = post_json(INGEST_URL, payload)
            predict_result = post_json(PREDICT_URL, payload)
            scored = (predict_result.get("results") or [{}])[0]

            print(status_line(transaction, intended_anomaly, scored))
            print(
                f"{Colors.DIM}  ingest:{ingest_result.get('status', 'unknown')} "
                f"predict:{predict_result.get('status', 'unknown')} "
                f"score:{scored.get('anomaly_score', 0):.4f}{Colors.RESET}"
            )

            if scored.get("is_anomaly"):
                print(
                    f"{Colors.YELLOW}  alert created: "
                    f"{scored.get('alert_id', 'n/a')}{Colors.RESET}"
                )
        except requests.exceptions.RequestException as exc:
            print(f"{Colors.RED}{Colors.BOLD}[ERROR]{Colors.RESET} {exc}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Simulator stopped by user.{Colors.RESET}")
            sys.exit(0)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
