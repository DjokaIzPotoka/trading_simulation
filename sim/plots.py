import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .params import Params


MEAN_COLOR = "#8B3FE6"    # ljubičasta
MEDIAN_COLOR = "#EF8D0D"  # narandžasta
START_COLOR = "#9E9E9E"   # siva


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

    plt.plot(x, mean_path, color=MEAN_COLOR, linewidth=2.5, label="Prosek (mean) kriva")
    plt.plot(x, median_path, color=MEDIAN_COLOR, linewidth=2.5, label="Mediana kriva")
    plt.axhline(p.start_capital, color=START_COLOR, linestyle="--", linewidth=2, label="Početni kapital")

    plt.scatter(T - 1, mean_final, color=MEAN_COLOR, s=60, label=f"Mean final: {mean_final:.2f}")
    plt.scatter(T - 1, median_final, color=MEDIAN_COLOR, s=60, label=f"Median final: {median_final:.2f}")

    seed_txt = "random" if seed is None else str(seed)
    plt.title(f"Simulacija balansa — {n_total} simulacija (seed={seed_txt}) | model={p.outcome_model}")
    plt.xlabel("Dani")
    plt.ylabel("Balans ($)")
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.xlim(0, p.br_day)
    plt.tight_layout()
    plt.legend()

    # === ZOOM oko mediane (percentile-based) ===
    low = np.percentile(paths, 1)
    high = np.percentile(paths, 99)

    plt.ylim(low, high)

    plt.show()


def plot_withdraw_histogram(df: pd.DataFrame, zoom_percentiles=(5, 95)):
    """
    Histogram total withdraw-a po simulaciji,
    sa percentile-based zoom-om (bez menjanja podataka).
    """
    data = df["total_withdraw"].to_numpy()

    # zoom granice
    low, high = np.percentile(data, zoom_percentiles)

    plt.figure(figsize=(11, 6))
    plt.hist(data, bins=60, range=(low, high), alpha=0.85)

    mean_w = float(np.mean(data))
    med_w = float(np.median(data))

    # mean / median linije (crtamo ih samo ako su u vidljivom opsegu)
    if low <= mean_w <= high:
        plt.axvline(mean_w, color="#8B3FE6", linewidth=2, label=f"Mean: {mean_w:.2f}")
    if low <= med_w <= high:
        plt.axvline(med_w, color="#EF8D0D", linewidth=2, label=f"Median: {med_w:.2f}")

    plt.title(f"Histogram: Total Withdraw (zoom {zoom_percentiles[0]}–{zoom_percentiles[1]} pct)")
    plt.xlabel("Total withdraw ($)")
    plt.ylabel("Broj simulacija")
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_profit_mean_median(equity_paths: np.ndarray,
                            withdraw_paths: np.ndarray,
                            p: Params,
                            n_total: int,
                            seed: int | None):
    """
    Profit(t) = equity(t) + cum_withdraw(t) - start_capital
    Plot samo mean i median krivu (bez svih simulacija).
    """
    profit_paths = equity_paths + withdraw_paths - p.start_capital

    T = profit_paths.shape[1]
    x = np.arange(T)

    mean_path = np.mean(profit_paths, axis=0)
    median_path = np.median(profit_paths, axis=0)

    mean_final = float(mean_path[-1])
    median_final = float(median_path[-1])

    plt.figure(figsize=(11, 6))

    plt.plot(x, mean_path, color=MEAN_COLOR, linewidth=2.8, label="Prosek (mean) profit kriva")
    plt.plot(x, median_path, color=MEDIAN_COLOR, linewidth=2.8, label="Mediana profit kriva")

    plt.axhline(0.0, color=START_COLOR, linestyle="--", linewidth=2, label="Breakeven (0)")

    plt.scatter(T - 1, mean_final, color=MEAN_COLOR, s=70, label=f"Mean final profit: {mean_final:.2f}")
    plt.scatter(T - 1, median_final, color=MEDIAN_COLOR, s=70, label=f"Median final profit: {median_final:.2f}")

    seed_txt = "random" if seed is None else str(seed)
    plt.title(f"Profit kroz vreme (mean/median) — {n_total} simulacija (seed={seed_txt}) | model={p.outcome_model}")
    plt.xlabel("Dani")
    plt.ylabel("Profit ($)")
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.xlim(0, p.br_day)
    plt.tight_layout()
    plt.legend()
    plt.show()



def plot_monthly_withdraw_bars(mean_by_month: np.ndarray, median_by_month: np.ndarray):
    """
    Bar chart:
      x = mesec (30-dnevni blok)
      y = withdraw ($)
      2 stuba po mesecu: mean i median
    """
    mean_by_month = np.asarray(mean_by_month, dtype=float)
    median_by_month = np.asarray(median_by_month, dtype=float)

    n_months = mean_by_month.size
    x = np.arange(1, n_months + 1)  # 1..n
    w = 0.35

    plt.figure(figsize=(11, 6))
    plt.bar(x - w/2, mean_by_month, width=w, alpha=0.85, label="Mean monthly withdraw")
    plt.bar(x + w/2, median_by_month, width=w, alpha=0.85, label="Median monthly withdraw")

    plt.title("Monthly Withdraw po mesecima (30-dnevni blokovi)")
    plt.xlabel("Mesec (blok od 30 dana)")
    plt.ylabel("Withdraw ($)")
    plt.xticks(x)
    plt.grid(True, linestyle="--", linewidth=0.6, axis="y")
    plt.tight_layout()
    plt.legend()
    plt.show()




