import numpy as np
from .params import Params


def generate_outcomes(n_trades: int, rng: np.random.Generator, p: Params):
    model = p.outcome_model.lower().strip()

    if model == "uniform":
        return _uniform_model(n_trades, rng, p)

    if model == "mixture_r":
        return _mixture_r_model(n_trades, rng, p)

    raise ValueError(f"Unknown outcome_model='{p.outcome_model}'. Use 'uniform' or 'mixture_r'.")


def _uniform_model(n_trades: int, rng: np.random.Generator, p: Params):
    wins = rng.random(n_trades) < p.winrate

    wp = (p.wp_base[0] * p.leverage, p.wp_base[1] * p.leverage)
    lp = (p.lp_base[0] * p.leverage, p.lp_base[1] * p.leverage)

    win_factors = rng.uniform(wp[0], wp[1], size=n_trades)
    loss_factors = rng.uniform(lp[0], lp[1], size=n_trades)

    multipliers = np.empty(n_trades, dtype=float)
    multipliers[wins] = win_factors[wins]
    multipliers[~wins] = -loss_factors[~wins]

    return multipliers, wins


def _mixture_r_model(n_trades: int, rng: np.random.Generator, p: Params):
    u = rng.random(n_trades)

    R = np.empty(n_trades, dtype=float)

    pL = p.p_full_loss
    pW = p.p_full_win
    if pL < 0 or pW < 0 or (pL + pW) > 1.0:
        raise ValueError("Invalid tail probabilities: require p_full_loss>=0, p_full_win>=0, p_full_loss+p_full_win<=1.")

    mask_full_loss = u < pL
    mask_full_win = (u >= pL) & (u < (pL + pW))
    mask_normal = ~(mask_full_loss | mask_full_win)

    R[mask_full_loss] = -1.0
    R[mask_full_win] = +1.0

    # Normal portion
    n_norm = int(mask_normal.sum())
    if n_norm > 0:
        u2 = rng.random(n_norm)
        norm_win = u2 < p.p_win_base

        R_norm = np.empty(n_norm, dtype=float)
        R_norm[norm_win] = rng.uniform(p.r_win_range[0], p.r_win_range[1], size=int(norm_win.sum()))
        R_norm[~norm_win] = -rng.uniform(p.r_loss_range[0], p.r_loss_range[1], size=int((~norm_win).sum()))

        R[mask_normal] = R_norm

    is_win = R > 0
    return R, is_win
