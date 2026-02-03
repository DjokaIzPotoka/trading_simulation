import numpy as np


def drawdown_stats(equity_curve: np.ndarray):
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return 0.0, 0.0, 0, 0.0

    running_max = np.maximum.accumulate(eq)
    dd_abs = running_max - eq

    denom = np.where(running_max > 0, running_max, np.nan)
    dd_pct = dd_abs / denom

    max_dd_abs = float(np.nanmax(dd_abs))
    max_dd_pct = float(np.nanmax(dd_pct)) if np.isfinite(np.nanmax(dd_pct)) else 0.0
    min_equity = float(np.min(eq))

    # drawdown duration: longest consecutive period where equity < running_max
    underwater = eq < running_max
    max_dur = 0
    cur = 0
    for u in underwater:
        if u:
            cur += 1
            if cur > max_dur:
                max_dur = cur
        else:
            cur = 0

    return max_dd_abs, max_dd_pct, int(max_dur), min_equity
