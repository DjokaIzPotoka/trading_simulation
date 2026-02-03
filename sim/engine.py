import numpy as np
from .params import Params
from .mechanics import calc_fee
from .outcomes import generate_outcomes
from .risk import drawdown_stats

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
    withdraw_events = []

    withdraw_curve = np.zeros(p.br_day + 1, dtype=float)
    cum_withdraw = 0.0
    withdraw_curve[0] = 0.0

    daily_pnls = np.zeros(p.br_day, dtype=float)
    equity_curve = np.zeros(p.br_day + 1, dtype=float)
    equity_curve[0] = capital

    n_trades = p.br_day * p.br_trade_day
    multipliers, is_win = generate_outcomes(n_trades, rng, p)

    t = 0
    for day in range(p.br_day):
        day_pnl = 0.0

        for _ in range(p.br_trade_day):
            trade = p.risk_percent * capital

            fee = calc_fee(trade, p)
            total_fee += fee

            pnl = (trade * multipliers[t]) - fee

            if is_win[t]:
                win_cnt += 1
                loss_streak = 0
                win_streak += 1
            else:
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

        if day_pnl > 0:
            green_days += 1
        else:
            red_days += 1

        # === EVENT-BASED WITHDRAW (daily check) ===
        if p.withdraw_mode == "event":
            if capital >= p.withdraw_trigger:
                excess = capital - p.withdraw_reset
                total_withdraw += excess
                cum_withdraw += excess
                withdraw_events.append((day, excess))
                capital = p.withdraw_reset

        equity_curve[day + 1] = capital
        withdraw_curve[day + 1] = cum_withdraw

    profit = total_withdraw + (capital - p.start_capital)

    max_dd_abs, max_dd_pct, max_dd_dur, min_equity = drawdown_stats(equity_curve)

    ruin_below_50 = int(min_equity < 50.0)
    ruin_below_20 = int(min_equity < 20.0)
    ruin_below_0  = int(min_equity < 0.0)

    months = max(1.0, p.br_day / 30.0)
    avg_monthly_withdraw_sim = total_withdraw / months

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

        "min_equity": float(min_equity),
        "max_dd_abs": float(max_dd_abs),
        "max_dd_pct": float(max_dd_pct),
        "max_dd_duration": int(max_dd_dur),

        "ruin_below_50": int(ruin_below_50),
        "ruin_below_20": int(ruin_below_20),
        "ruin_below_0": int(ruin_below_0),

        "total_fee": float(total_fee),

        "total_withdraw": float(total_withdraw),
        "profit": float(profit),

        "avg_withdraw_event": float(np.mean(withdraw_events)) if withdraw_events else 0.0,
        "median_withdraw_event": float(np.median(withdraw_events)) if withdraw_events else 0.0,
        "withdraw_event_count": int(len(withdraw_events)),
        "avg_monthly_withdraw_sim": float(avg_monthly_withdraw_sim),
        "median_monthly_withdraw_sim": 0.0,

    }

    return stats, equity_curve, withdraw_curve, np.array(withdraw_events, dtype=object)
