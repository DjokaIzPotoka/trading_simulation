from .params import Params


def calc_fee(trade: float, p: Params) -> float:
    # EXACT per your definition:
    # trade * (leverage*100) * fee_rate * 2.06
    return trade * (p.leverage * 100.0) * p.fee_rate * p.fee_multiplier
