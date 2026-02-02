import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    start_capital: float = 100.0
    winrate: float = 0.67
    leverage: float = 0.50

    # Uniform multipliers are (base_range * leverage)
    wp_base: tuple = (0.6, 0.7)
    lp_base: tuple = (0.6, 0.7)

    risk_percent: float = 0.25

    br_day: int = 20
    br_trade_day: int = 2

    # fee: trade * (leverage*100) * fee_rate * 2.06
    fee_rate: float = 0.0005
    fee_multiplier: float = 2.06

    # withdraw: every 30 days, withdraw everything above target, leave target
    withdraw_every_days: int = 30
    withdraw_target: float = 2000.0

    # plotting
    plot_paths_max: int = 400


def calc_fee(trade: float, p: Params) -> float:
    # EXACT per your definition:
    return trade * (p.leverage * 100.0) * p.fee_rate * p.fee_multiplier


def simulate_one(rng: np.random.Generator, p: Params):
    capital = float(p.start_capital)

    win_cnt = 0
    loss_cnt = 0

    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0

    green_days = 0
    red_days = 0

    total_fee = 0.0
    total_withdraw = 0.0
    monthly_withdraws = []

    daily_pnls = np.zeros(p.br_day, dtype=float)
    equity_curve = np.zeros(p.br_day + 1, dtype=float)
    equity_curve[0] = capital

    # Pre-generate randomness (fast + correct)
    n_trades = p.br_day * p.br_trade_day
    wins = rng.random(n_trades) < p.winrate

    wp = (p.wp_base[0] * p.leverage, p.wp_base[1] * p.leverage)
    lp = (p.lp_base[0] * p.leverage, p.lp_base[1] * p.leverage)
    win_factors = rng.uniform(wp[0], wp[1], size=n_trades)
    loss_factors = rng.uniform(lp[0], lp[1], size=n_trades)

    t = 0
    for day in range(p.br_day):
        day_pnl = 0.0

        for _ in range(p.br_trade_day):
            trade = p.risk_percent * capital

            fee = calc_fee(trade, p)
            total_fee += fee

            if wins[t]:
                pnl = (trade * win_factors[t]) - fee
                win_cnt += 1
                loss_streak = 0
                win_streak += 1
            else:
                pnl = (-trade * loss_factors[t]) - fee
                loss_cnt += 1
                win_streak = 0
                loss_streak += 1

            if win_streak > max_win_streak:
                max_win_streak = win_streak
            if loss_streak > max_loss_streak:
                max_loss_streak = loss_streak

            capital += pnl
            day_pnl += pnl
            t += 1

        daily_pnls[day] = day_pnl
        equity_curve[day + 1] = capital

        if day_pnl > 0:
            green_days += 1
        else:
            red_days += 1

        # Month boundary withdraw
        if (day + 1) % p.withdraw_every_days == 0:
            if capital >= p.withdraw_target:
                excess = capital - p.withdraw_target
                total_withdraw += excess
                monthly_withdraws.append(excess)
                capital = p.withdraw_target
                # reflect reset in chart
                equity_curve[day + 1] = capital
            else:
                monthly_withdraws.append(0.0)

    profit = total_withdraw + (capital - p.start_capital)

    stats = {
        "final_capital": float(capital),
        "win_cnt": int(win_cnt),
        "loss_cnt": int(loss_cnt),
        "br_trade": int(p.br_day * p.br_trade_day),
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),

        "avg_day_pnl": float(daily_pnls.mean()),
        "total_pnl": float(daily_pnls.sum()),
        "green_days": int(green_days),
        "red_days": int(red_days),
        "max_day_pnl": float(daily_pnls.max()),
        "min_day_pnl": float(daily_pnls.min()),

        "total_fee": float(total_fee),

        "total_withdraw": float(total_withdraw),
        "profit": float(profit),

        # per-simulation monthly summaries (6 values for 180 days @ 30d)
        "avg_monthly_withdraw_sim": float(np.mean(monthly_withdraws)) if monthly_withdraws else 0.0,
        "median_monthly_withdraw_sim": float(np.median(monthly_withdraws)) if monthly_withdraws else 0.0,
    }

    return stats, equity_curve, np.array(monthly_withdraws, dtype=float)


def monte_carlo(n: int, p: Params, seed: int | None = None):
    # seed=None -> non-deterministic runs (different every program run)
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

    # === Requested aggregates ===

    # Total withdraw across sims
    mean_total_withdraw = float(df["total_withdraw"].mean())
    median_total_withdraw = float(df["total_withdraw"].median())

    # Monthly withdraw pooled (all sims x all months)
    mean_monthly_pooled = float(pooled_monthly_withdraws.mean()) if pooled_monthly_withdraws.size else 0.0
    median_monthly_pooled = float(np.median(pooled_monthly_withdraws)) if pooled_monthly_withdraws.size else 0.0

    # Per-sim monthly avg, aggregated across sims
    mean_monthly_per_sim = float(df["avg_monthly_withdraw_sim"].mean())
    median_monthly_per_sim = float(df["avg_monthly_withdraw_sim"].median())

    # Fees across sims
    mean_fee = float(df["total_fee"].mean())
    median_fee = float(df["total_fee"].median())

    # Profit across sims
    mean_profit = float(df["profit"].mean())
    median_profit = float(df["profit"].median())

    pct_sims_with_withdraw = float((df["total_withdraw"] > 0).mean() * 100.0)
    pct_sims_profit_pos = float((df["profit"] > 0).mean() * 100.0)
    pct_months_with_withdraw = float((pooled_monthly_withdraws > 0).mean() * 100.0) if pooled_monthly_withdraws.size else 0.0


    print("\n===== AGREGAT (SVE SIMULACIJE) =====")
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


def plot_spaghetti(paths: np.ndarray, p: Params, n_total: int, seed: int | None):
    n, T = paths.shape
    x = np.arange(T)

    mean_path = np.mean(paths, axis=0)
    median_path = np.median(paths, axis=0)

    mean_final = float(mean_path[-1])
    median_final = float(median_path[-1])

    plot_n = min(n, p.plot_paths_max)
    idx = np.linspace(0, n - 1, plot_n).astype(int)
    cols = plt.cm.Blues(np.linspace(0.35, 0.95, plot_n))

    plt.figure(figsize=(11, 6))
    for k, i in enumerate(idx):
        plt.plot(x, paths[i], linewidth=1, alpha=0.65, color=cols[k])

    plt.plot(
        x,
        mean_path,
        color="#8B3FE6",  # ljubičasta
        linewidth=2.5,
        label="Prosek (mean) kriva",
    )

    plt.plot(
        x,
        median_path,
        color="#EF8D0D",  # narandžasta
        linewidth=2.5,
        label="Mediana kriva",
    )

    plt.axhline(
        p.start_capital,
        color="#9E9E9E",  # siva
        linestyle="--",
        linewidth=2,
        label="Početni kapital",
    )

    plt.scatter(T - 1, mean_final, s=60, label=f"Mean final: {mean_final:.2f}")
    plt.scatter(T - 1, median_final, s=60, label=f"Median final: {median_final:.2f}")

    seed_txt = "random" if seed is None else str(seed)
    plt.title(f"Simulacija balansa — {n_total} simulacija (seed={seed_txt})")
    plt.xlabel("Dani")
    plt.ylabel("Balans ($)")
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.xlim(0, p.br_day)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_withdraw_histogram(df: pd.DataFrame):
    """
    Histogram ukupnog withdraw-a po simulaciji.
    Distribucija je obično right-skewed (tail), pa ovo lepo objasni mean vs median.
    """
    data = df["total_withdraw"].to_numpy()

    plt.figure(figsize=(11, 6))
    plt.hist(data, bins=60, alpha=0.85)

    mean_w = float(np.mean(data))
    med_w = float(np.median(data))

    plt.axvline(mean_w, linewidth=2, label=f"Mean: {mean_w:.2f}")
    plt.axvline(med_w, linewidth=2, label=f"Median: {med_w:.2f}")

    plt.title("Histogram: Total Withdraw (po simulaciji)")
    plt.xlabel("Total withdraw ($)")
    plt.ylabel("Broj simulacija")
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.legend()
    plt.show()


def main():
    p = Params(
        start_capital=100.0,
        winrate=0.67,
        leverage=0.50,
        wp_base=(0.6, 0.7),
        lp_base=(0.6, 0.7),
        risk_percent=0.25,
        br_day=180,
        br_trade_day=2,
        fee_rate=0.0005,
        fee_multiplier=2.06,
        withdraw_every_days=30,
        withdraw_target=2000.0,
        plot_paths_max=800,
    )

    n = 1000

    # IMPORTANT:
    # seed=None => svaki put kad pokreneš program, dobijaš drugačiji rezultat
    # seed=42  => reproducible (uvek isti rezultat za taj seed)
    seed = None

    df, paths = monte_carlo(n, p, seed=seed)

    df.round(2).to_csv("stats.csv", index=False)
    print("Sačuvan stats.csv")

    plot_spaghetti(paths, p, n_total=n, seed=seed)
    plot_withdraw_histogram(df)



if __name__ == "__main__":
    main()
