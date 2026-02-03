import numpy as np
import pandas as pd
from .params import Params
from .engine import simulate_one


def monte_carlo(n: int, p: Params, seed: int | None = None):
    rng = np.random.default_rng(seed)

    results = []
    paths = np.zeros((n, p.br_day + 1), dtype=float)
    withdraw_paths = np.zeros((n, p.br_day + 1), dtype=float)

    monthly_withdraws_all = []  # LISTA dok skupljamo

    n_months = int(np.ceil(p.br_day / 30))
    monthly_matrix = np.zeros((n, n_months), dtype=float)

    for i in range(n):
        stats, curve, wcurve, withdraw_events = simulate_one(rng, p)

        # withdraw_events: [(day, amount), ...]
        months = {}
        for day, amount in withdraw_events:
            month = day // 30
            months[month] = months.get(month, 0.0) + float(amount)

        for m in range(n_months):
            val = months.get(m, 0.0)
            monthly_matrix[i, m] = val
            monthly_withdraws_all.append(val)

        results.append(stats)
        paths[i] = curve
        withdraw_paths[i] = wcurve

    monthly_withdraws_all = np.array(monthly_withdraws_all, dtype=float)

    df = pd.DataFrame(results)

    mean_withdraw_by_month = np.mean(monthly_matrix, axis=0)
    median_withdraw_by_month = np.median(monthly_matrix, axis=0)

    # Aggregates
    mean_total_withdraw = float(df["total_withdraw"].mean())
    median_total_withdraw = float(df["total_withdraw"].median())

    mean_monthly_pooled = float(monthly_withdraws_all.mean()) if monthly_withdraws_all.size else 0.0
    median_monthly_pooled = float(np.median(monthly_withdraws_all)) if monthly_withdraws_all.size else 0.0

    mean_monthly_per_sim = float(df["avg_withdraw_event"].mean())
    median_monthly_per_sim = float(df["avg_withdraw_event"].median())

    mean_fee = float(df["total_fee"].mean())
    median_fee = float(df["total_fee"].median())

    mean_profit = float(df["profit"].mean())
    median_profit = float(df["profit"].median())

    # === PROFIT percentiles ===
    p5_profit = float(df["profit"].quantile(0.05))
    p50_profit = float(df["profit"].quantile(0.50))
    p95_profit = float(df["profit"].quantile(0.95))

    # === Drawdown percentiles ===
    p50_dd_pct = float(df["max_dd_pct"].quantile(0.50))
    p95_dd_pct = float(df["max_dd_pct"].quantile(0.95))
    p50_dd_abs = float(df["max_dd_abs"].quantile(0.50))
    p95_dd_abs = float(df["max_dd_abs"].quantile(0.95))

    # === Ruin probabilities ===
    ruin50 = float(df["ruin_below_50"].mean() * 100.0)
    ruin20 = float(df["ruin_below_20"].mean() * 100.0)
    ruin0 = float(df["ruin_below_0"].mean() * 100.0)

    pct_sims_with_withdraw = float((df["total_withdraw"] > 0).mean() * 100.0)
    pct_sims_profit_pos = float((df["profit"] > 0).mean() * 100.0)
    pct_months_with_withdraw = float((monthly_withdraws_all > 0).mean() * 100.0) if monthly_withdraws_all.size else 0.0

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

    print("\n--- PROFIT PERCENTILES ---")
    print(f"P5 profit   : {p5_profit:.2f}")
    print(f"P50 profit  : {p50_profit:.2f}")
    print(f"P95 profit  : {p95_profit:.2f}")

    print("\n--- DRAWDOWN (MAX) PERCENTILES ---")
    print(f"P50 max DD %   : {p50_dd_pct*100:.2f}%")
    print(f"P95 max DD %   : {p95_dd_pct*100:.2f}%")
    print(f"P50 max DD $   : {p50_dd_abs:.2f}")
    print(f"P95 max DD $   : {p95_dd_abs:.2f}")

    print("\n--- RUIN PROBABILITY (min equity below) ---")
    print(f"Below 50 : {ruin50:.2f}%")
    print(f"Below 20 : {ruin20:.2f}%")
    print(f"Below 0  : {ruin0:.2f}%")

    print("\n--- FEES ---")
    print(f"Mean total fee   : {mean_fee:.2f}")
    print(f"Median total fee : {median_fee:.2f}")

    print("\n--- FINAL CAPITAL ---")
    print(f"Median final capital : {df['final_capital'].median():.2f}")
    print(f"Mean final capital   : {df['final_capital'].mean():.2f}")

    return df, paths, withdraw_paths, monthly_withdraws_all, mean_withdraw_by_month, median_withdraw_by_month


def export_csv(df: pd.DataFrame, path: str = "stats.csv"):
    df.round(2).to_csv(path, index=False)
    print(f"Sačuvan {path}")
