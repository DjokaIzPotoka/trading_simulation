from sim import Params, monte_carlo, export_csv, plot_spaghetti, plot_withdraw_histogram
from sim.plots import plot_profit_mean_median, plot_monthly_withdraw_bars


def main():
    p = Params(
        start_capital=108.0,
        br_day=180,
        br_trade_day=2,
        risk_percent=0.1,

        leverage=0.50,
        fee_rate=0.0005,
        fee_multiplier=2.06,

        withdraw_every_days=30,
        withdraw_target=6000.0,
        withdraw_trigger=6000,
        withdraw_reset = 2000,

        plot_paths_max=800,

        # choose:
        outcome_model="mixture_r",  # "uniform"
        withdraw_mode="event",

        # uniform model params
        winrate=0.67,
        wp_base=(0.6, 0.7),
        lp_base=(0.6, 0.7),

        # mixture model params (used only if outcome_model="mixture_r")
        p_full_loss=0.013,
        p_full_win=0.005,
        p_win_base=0.67,
        r_win_range=(0.5, 0.65),
        r_loss_range=(0.65, 0.75),
    )

    n = 2000
    seed = None

    df, equity_paths, withdraw_paths, monthly_withdraws, mean_by_month, median_by_month = monte_carlo(n, p, seed=seed)
    export_csv(df, "stats.csv")
    plot_spaghetti(equity_paths, p, n_total=n, seed=seed)
    plot_withdraw_histogram(df)
    plot_profit_mean_median(equity_paths, withdraw_paths, p, n_total=n, seed=seed)
    plot_monthly_withdraw_bars(mean_by_month, median_by_month)


if __name__ == "__main__":
    main()
