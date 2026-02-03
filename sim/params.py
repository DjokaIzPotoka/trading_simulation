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

    # ===== event-based withdraw =====
    withdraw_mode: str = "event"

    withdraw_trigger: float = 5000.0
    withdraw_reset: float = 2000.0


    # plotting
    plot_paths_max: int = 800

    # ===== outcome model selection =====

    outcome_model: str = "uniform"

    # ===== uniform model params
    winrate: float = 0.67
    wp_base: tuple = (0.6, 0.7)
    lp_base: tuple = (0.6, 0.7)

    # ===== mixture R-multiple model params
    p_full_loss: float = 0.005
    p_full_win: float = 0.002

    # For normal trades
    p_win_base: float = 0.67

    # Normal R ranges
    r_win_range: tuple = (0.20, 0.35)
    r_loss_range: tuple = (0.20, 0.35)
