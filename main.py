from sim import Params, monte_carlo, export_csv, plot_spaghetti, plot_withdraw_histogram


def main():
    p = Params(
        start_capital=100.0,
        br_day=180,
        br_trade_day=2,
        risk_percent=0.25,

        leverage=0.50,
        fee_rate=0.0005,
        fee_multiplier=2.06,

        withdraw_every_days=30,
        withdraw_target=2000.0,

        plot_paths_max=1000,

        # choose:
        outcome_model="mixture_r",  # <-- promeni u "uniform" ako hoćeš stari model

        # uniform model params (used only if outcome_model="uniform")
        winrate=0.67,
        wp_base=(0.6, 0.7),
        lp_base=(0.6, 0.7),

        # mixture model params (used only if outcome_model="mixture_r")
        p_full_loss=0.02,   # 0.5% full -1R
        p_full_win=0.005,    # 0.2% full +1R
        p_win_base=0.67,     # win chance for normal trades
        r_win_range=(0.5, 0.65),
        r_loss_range=(0.65, 0.75),
    )

    n = 2000
    seed = None  # None => svaki run drugačiji, 42 => reproducible

    df, paths = monte_carlo(n, p, seed=seed)
    export_csv(df, "stats.csv")

    plot_spaghetti(paths, p, n_total=n, seed=seed)
    plot_withdraw_histogram(df)


if __name__ == "__main__":
    main()
