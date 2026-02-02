from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # capital & schedule
    start_capital: float = 100.0
    br_day: int = 180
    br_trade_day: int = 2

    # sizing
    risk_percent: float = 0.25

    # leverage & fee
    leverage: float = 0.50
    fee_rate: float = 0.0005
    fee_multiplier: float = 2.06

    # withdraw
    withdraw_every_days: int = 30
    withdraw_target: float = 2000.0

    # plotting
    plot_paths_max: int = 800

    # ===== outcome model selection =====
    # "uniform" = tvoj trenutni model (winrate + uniform ranges)
    # "mixture_r" = realistic scenario model (tail +/-1R + normal R distribution)
    outcome_model: str = "uniform"

    # ===== uniform model params (tvoj trenutni) =====
    winrate: float = 0.67
    wp_base: tuple = (0.6, 0.7)  # multiplied by leverage
    lp_base: tuple = (0.6, 0.7)  # multiplied by leverage

    # ===== mixture R-multiple model params (novi) =====
    # Tail events: % trades that are full -1R or +1R
    p_full_loss: float = 0.005   # 0.5% trades -> -1.0R
    p_full_win: float = 0.002    # 0.2% trades -> +1.0R

    # For normal trades (when not tail), chance of win
    p_win_base: float = 0.67

    # Normal R ranges (in units of "risk" = trade size)
    r_win_range: tuple = (0.20, 0.35)   # +0.20R .. +0.35R
    r_loss_range: tuple = (0.20, 0.35)  # -0.20R .. -0.35R
