import numpy as np
import matplotlib.pyplot as plt

#===========================
#POCETNI PARAMETRI ZA TRADE
#===========================

start_capital = 100
winrate = 0.40
lavrage = 0.25
wp = (0.4 * lavrage, 0.95 * lavrage)
lp = (0.2 * lavrage, 0.5 * lavrage)
risk_precent = 0.1
trade = 10
fee_rate = 0.0005
br_day = 30
br_trade_day = 5
br_trade = br_trade_day * br_day

pos_size = trade * lavrage

def simulate(start_capital, winrate, lavrage, trade, pos_size, br_trade):
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
    return stats

print(simulate(start_capital, winrate, lavrage, trade, pos_size, br_trade))