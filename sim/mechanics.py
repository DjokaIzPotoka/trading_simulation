from .params import Params


def calc_fee(trade: float, p: Params) -> float:
    return trade * (p.leverage * 100.0) * p.fee_rate * p.fee_multiplier
