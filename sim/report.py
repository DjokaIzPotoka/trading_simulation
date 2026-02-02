import numpy as np
import pandas as pd
from .params import Params
from .engine import simulate_one


def monte_carlo(n: int, p: Params, seed: int | None = None):
    rng = np.random.default_rng(seed)

    results = []
    paths = np.zeros((n, p.br_day + 1), dtype=float)
    pooled_monthly_withdraws = []

    for i in range(n):
        stats, curve, monthly_w = simulate_one(rng, p)
        results.append(stats)
        paths[i] = curve
        pooled_monthly_withdraws.extend(monthly_w.tolist())

    df = pd.DataFrame(results)
    pooled_monthly_withdraws = np.array(pooled_monthly_withdraws, dtype=float)

    # Aggregates
    mean_total_withdraw = float(df["total_withdraw"].mean())
    median_total_withdraw = float(df["total_withdraw"].median())

    mean_monthly_pooled = float(pooled_monthly_withdraws.mean()) if pooled_monthly_withdraws.size else 0.0
    median_monthly_pooled = float(np.median(pooled_monthly_withdraws)) if pooled_monthly_withdraws.size else 0.0

    mean_monthly_per_sim = float(df["avg_monthly_withdraw_sim"].mean())
    median_monthly_per_sim = float(df["avg_monthly_withdraw_sim"].median())

    mean_fee = float(df["total_fee"].mean())
    median_fee = float(df["total_fee"].median())

    mean_profit = float(df["profit"].mean())
    median_profit = float(df["profit"].median())

    pct_sims_with_withdraw = float((df["total_withdraw"] > 0).mean() * 100.0)
    pct_sims_profit_pos = float((df["profit"] > 0).mean() * 100.0)
    pct_months_with_withdraw = float((pooled_monthly_withdraws > 0).mean() * 100.0) if pooled_monthly_withdraws.size else 0.0

    print("\n===== AGREGAT (SVE SIMULACIJE) =====")
    print(f"Outcome model: {p.outcome_model}")
    print(f"Daily average PnL (mean over sims): {df['avg_day_pnl'].mean():.4f}")

    print("\n--- WITHDRAW (TOTAL per sim) ---")
    print(f"Mean total withdraw   : {mean_total_withdraw:.2f}")
    print(f"Median total withdraw : {median_total_withdraw:.2f}")

    print("\n--- WITHDRAW (MONTHLY) ---")
    print(f"Mean monthly withdraw (pooled all months)   : {mean_monthly_pooled:.2f}")
    print(f"Median monthly withdraw (pooled all months) : {median_monthly_pooled:.2f}")
    print(f"Mean monthly withdraw (per-sim avg)         : {mean_monthly_per_sim:.2f}")
    print(f"Median monthly withdraw (per-sim avg)       : {median_monthly_per_sim:.2f}")

    print("\n--- PROFIT (withdraw + final - start) ---")
    print(f"Mean profit   : {mean_profit:.2f}")
    print(f"Median profit : {median_profit:.2f}")

    print("\n--- HIT RATE / SUCCESS METRICS ---")
    print(f"% simulations with >=1 withdraw : {pct_sims_with_withdraw:.2f}%")
    print(f"% simulations with profit > 0   : {pct_sims_profit_pos:.2f}%")
    print(f"% months with withdraw > 0      : {pct_months_with_withdraw:.2f}%")

    print("\n--- FEES ---")
    print(f"Mean total fee   : {mean_fee:.2f}")
    print(f"Median total fee : {median_fee:.2f}")

    print("\n--- FINAL CAPITAL ---")
    print(f"Median final capital : {df['final_capital'].median():.2f}")
    print(f"Mean final capital   : {df['final_capital'].mean():.2f}")

    return df, paths


def export_csv(df: pd.DataFrame, path: str = "stats.csv"):
    df.round(2).to_csv(path, index=False)
    print(f"Sačuvan {path}")
