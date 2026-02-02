import numpy as np
from .params import Params
from .mechanics import calc_fee
from .outcomes import generate_outcomes


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

        "avg_monthly_withdraw_sim": float(np.mean(monthly_withdraws)) if monthly_withdraws else 0.0,
        "median_monthly_withdraw_sim": float(np.median(monthly_withdraws)) if monthly_withdraws else 0.0,
    }

    return stats, equity_curve, np.array(monthly_withdraws, dtype=float)
