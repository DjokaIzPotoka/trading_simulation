import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from fontTools.misc.cython import returns
from matplotlib import colors
from matplotlib.colors import LogNorm
from numpy.ma.extras import median

#===========================
#POCETNI PARAMETRI ZA TRADE
#===========================

start_capital = 100
winrate = 0.40
lavrage = 0.25
wp = (0.4 * lavrage, 0.95 * lavrage)
lp = (0.2 * lavrage, 0.5 * lavrage)
risk_precent = 0.15
trade = 10
fee_rate = 0.0005
br_day = 30
br_trade_day = 10
br_trade = br_trade_day * br_day

pos_size = trade * lavrage

def simulate(start_capital):
    win_cnt = 0
    loss_cnt = 0
    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    day_pnl = 0
    green = 0
    red = 0

    daily_pnls = []
    equity_curve = [start_capital]

    i = 0

    while i < br_day:
        for j in range (0, br_trade_day):
            trade = risk_precent * start_capital
            win = np.random.rand() < winrate
            if win:
                pnl_gross = trade + trade * np.random.uniform(*wp)
                win_cnt +=1
                loss_streak = 0
                win_streak += 1
            else:
                win_streak = 0
                loss_cnt += 1
                pnl_gross = trade - trade * np.random.uniform(*lp)
                loss_streak +=1

            fee = trade * fee_rate
            pnl = (pnl_gross - trade) - fee
            start_capital += pnl
            day_pnl += pnl

            if win_streak > max_win_streak:
                max_win_streak = win_streak
            if loss_streak > max_loss_streak:
                max_loss_streak = loss_streak

        daily_pnls.append(day_pnl)
        equity_curve.append(start_capital)

        if day_pnl > 0:
            green += 1
        else:
            red += 1

        day_pnl = 0
        i+=1

    total_pnl = sum(daily_pnls)
    average_day_pnl = total_pnl / br_day
    max_day_pnl = max(daily_pnls)
    min_day_pnl = min(daily_pnls)

    stats = {

        "capital": start_capital,
        "win_cnt": win_cnt,
        "loss_cnt": loss_cnt,
        "br_trade": br_trade,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "average_day_pnl": average_day_pnl,
        "green": green,
        "red": red,
        "max_day_pnl": max_day_pnl,
        "min_day_pnl": min_day_pnl,

    }
    print(stats)
    return stats, equity_curve


def monteCarlo(n):
    results = []
    all_paths = []
    for i in range(n):
        stats, paths = simulate(start_capital)
        results.append(stats)
        all_paths.append(paths)

    return results, np.array(all_paths)

def main():
    n = 5000
    results, paths = monteCarlo(n)
    last_x = len(paths[0]) - 1

    plt.figure(figsize=(11, 6))
    colors = plt.cm.Blues(np.linspace(0.3, 1, n))

    final_capitals = [p[-1] for p in paths]

    mean_path = np.mean(paths, axis=0)
    median_path = np.median(paths, axis=0)
    mean_final = np.mean(final_capitals)
    median_final = np.median(final_capitals)

    num_paths, num_points = paths.shape
    x = np.arange(num_points)
    num_fine = num_points * 50

    x_fine = np.linspace(x.min(), x.max(), num_fine)
    # x_all = np.broadcast_to(x, (num_paths, num_points)).ravel()
    y_all = paths.ravel()
    y_fine = np.concatenate([np.interp(x_fine, x, row) for row in paths])
    x_fine_all = np.broadcast_to(x_fine, (num_paths, num_fine)).ravel()

    h, xedges, yedges = np.histogram2d(x_fine_all, y_fine, bins=[num_fine, 250])

    for i, s in enumerate(paths):
        plt.plot(s, linewidth=1, color=colors[i], alpha=0.7)

    plt.plot(mean_path, color="#BA3CE1", linewidth=2, label="Prosek kriva")
    plt.axhline(start_capital, color="red", linestyle="--", linewidth=2, label="Pocetni kapital")
    plt.plot(median_path, color="#EF8D0D", linewidth=2, label="Mediana kriva")
    plt.title(f"Simulacija balansa — {n} simulacija")
    plt.xlabel("Dani")
    plt.ylabel("Balans ($)")
    plt.grid(True, linestyle='--', linewidth=0.6)
    plt.tight_layout()
    plt.xlim(0, br_day)
    plt.scatter(last_x, mean_final, color="purple", s=50, label=f"Prosek: {mean_final:.2f}")
    plt.scatter(last_x, median_final, color="orange", s=50, label=f"Mediana: {median_final:.2f}")
    plt.legend()

    plt.show()

    df = pd.DataFrame(results)  # napravi tabelu iz liste stats dict-ova
    df_rounded = df.round(2)  # zaokruži sve numeričke kolone na 2 decimale

    plt.clf()
    plt.close()



    plt.figure(figsize=(11, 6))
    pcm = plt.pcolormesh(xedges, yedges, h.T, cmap="inferno", norm=LogNorm(vmin=1e0, vmax=150))
    plt.title("Heatmap gustine - {n} simulacija}")
    plt.axhline(start_capital, color="black", linestyle="--", linewidth=2, label="Pocetni kapital")
    plt.xlabel("Dani")
    plt.ylabel("Balans ($)")
    plt.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.6)
    plt.xlim(0, br_day)

    plt.show()

    df_rounded.to_csv("stats.csv", index=False)  # export u CSV
    print("Sačuvan u stats.csv")


if __name__ == "__main__":
    main()

