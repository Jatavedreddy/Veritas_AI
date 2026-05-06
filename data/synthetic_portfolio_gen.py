import pandas as pd
import numpy as np
from pathlib import Path
import os

def generate_portfolios():
    print("Generating synthetic portfolio data...")
    
    # Define client profiles
    clients = [
        {"id": "C001", "name": "Sterling Venture Partners", "profile": "Aggressive Growth"},
        {"id": "C002", "name": "Arthur J. Henderson (Private)", "profile": "Income Preservation"},
        {"id": "C003", "name": "Lumina Digital Assets", "profile": "Extreme Risk"},
        {"id": "C004", "name": "St. Jude Foundation Trust", "profile": "Balanced"},
        {"id": "C005", "name": "Green Future Endowment", "profile": "Sustainable Growth"}
    ]
    
    # Asset universe
    assets = [
        # Equities - Tech
        {"class": "Equity", "ticker": "NVDA", "sector": "Technology", "beta": 1.8, "esg": 65},
        {"class": "Equity", "ticker": "AAPL", "sector": "Technology", "beta": 1.2, "esg": 72},
        {"class": "Equity", "ticker": "MSFT", "sector": "Technology", "beta": 1.1, "esg": 85},
        # Equities - Traditional
        {"class": "Equity", "ticker": "JPM", "sector": "Financials", "beta": 1.0, "esg": 60},
        {"class": "Equity", "ticker": "XOM", "sector": "Energy", "beta": 0.8, "esg": 35},
        {"class": "Equity", "ticker": "JNJ", "sector": "Healthcare", "beta": 0.6, "esg": 75},
        # Fixed Income
        {"class": "Fixed Income", "ticker": "US10Y", "sector": "Government", "beta": 0.0, "esg": 50},
        {"class": "Fixed Income", "ticker": "LQD", "sector": "Corporate Bonds", "beta": 0.2, "esg": 65},
        {"class": "Fixed Income", "ticker": "MUB", "sector": "Municipal", "beta": 0.1, "esg": 80},
        # Crypto
        {"class": "Crypto", "ticker": "BTC", "sector": "Digital Asset", "beta": 3.0, "esg": 20},
        {"class": "Crypto", "ticker": "ETH", "sector": "Digital Asset", "beta": 3.5, "esg": 40},
        # Alternatives
        {"class": "Real Estate", "ticker": "VNQ", "sector": "REIT", "beta": 0.9, "esg": 60},
        {"class": "Commodity", "ticker": "GLD", "sector": "Precious Metals", "beta": -0.1, "esg": 50},
        {"class": "Equity", "ticker": "ICLN", "sector": "Clean Energy", "beta": 1.4, "esg": 95}
    ]
    
    data = []
    
    for client in clients:
        # Determine holdings based on profile
        if client["profile"] == "Aggressive Growth":
            weights = [("NVDA", 0.3), ("AAPL", 0.2), ("MSFT", 0.3), ("BTC", 0.1), ("ICLN", 0.1)]
            base_capital = 5000000
        elif client["profile"] == "Income Preservation":
            weights = [("US10Y", 0.4), ("LQD", 0.3), ("MUB", 0.1), ("JNJ", 0.15), ("GLD", 0.05)]
            base_capital = 2000000
        elif client["profile"] == "Extreme Risk":
            weights = [("BTC", 0.5), ("ETH", 0.3), ("NVDA", 0.2)]
            base_capital = 500000
        elif client["profile"] == "Balanced":
            weights = [("AAPL", 0.15), ("MSFT", 0.15), ("JPM", 0.1), ("US10Y", 0.3), ("LQD", 0.2), ("VNQ", 0.1)]
            base_capital = 10000000
        elif client["profile"] == "Sustainable Growth":
            weights = [("ICLN", 0.4), ("MSFT", 0.2), ("JNJ", 0.2), ("MUB", 0.2)]
            base_capital = 3000000
            
        for ticker, weight in weights:
            asset = next(a for a in assets if a["ticker"] == ticker)
            invested = base_capital * weight
            
            # Simulate some return (-15% to +40% depending on beta)
            np.random.seed(hash(client["id"] + ticker) % (2**32))
            return_pct = np.random.normal(0.05, 0.1 * abs(asset["beta"]))
            current_value = invested * (1 + return_pct)
            
            data.append({
                "Client_ID": client["id"],
                "Client_Name": client["name"],
                "Profile": client["profile"],
                "Asset_Class": asset["class"],
                "Ticker": asset["ticker"],
                "Sector": asset["sector"],
                "Invested_Amount": round(invested, 2),
                "Current_Value": round(current_value, 2),
                "Return_Pct": round(return_pct * 100, 2),
                "Risk_Beta": asset["beta"],
                "ESG_Score": asset["esg"]
            })
            
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_dir = Path(__file__).parent / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "portfolios.csv"
    
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df)} portfolio holdings records.")
    print(f"✅ Saved to: {out_path}")
    print("\nYou can now load this CSV into Power BI Desktop for your Portfolio Insights dashboard!")

if __name__ == "__main__":
    generate_portfolios()
