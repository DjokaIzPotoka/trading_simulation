import random
import matplotlib.pyplot as plt

#POCETNI PARAMETRI ZA TRADE
start_capital = 100
winrate = 0.44
lavrage = 0.25
trade = 10
br_trade = 100
pos_size = trade * lavrage

def simulate(start_capital, winrate, lavrage, trade, pos_size, br_trade):
    i = 0
    win_cnt = 0
    loss_cnt = 0
    while i < br_trade:
        win = random.random() < winrate
        if win:
            pnl_gross = trade + trade * random.uniform(0.4 * lavrage, 0.9 * lavrage)
            win_cnt +=1
        else:
            pnl_gross = trade - trade * random.uniform(0.2 * lavrage, 0.5 * lavrage)
            loss_cnt +=1

        pnl = pnl_gross - trade
        start_capital += pnl
        i+=1

    stats = {
        "capital": start_capital,
        "win_cnt": win_cnt,
        "loss_cnt": loss_cnt,
    }
    return stats

print(simulate(start_capital, winrate, lavrage, trade, pos_size, br_trade))