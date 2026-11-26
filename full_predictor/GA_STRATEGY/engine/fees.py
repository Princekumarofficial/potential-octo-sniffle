"""Current FYERS Standard NSE equity intraday tariff, checked 2026-11-16.

Sources: https://fyers.in/charges-list and https://fyers.in/pricing
Amounts are calculated per executed order and rounded to paise. Contract-note
aggregation/rounding may differ. These are current-rate scenario estimates.
"""
COMPONENTS = ("brokerage", "stt", "exchange", "sebi", "ipft", "stamp", "gst", "admin", "dp")


def intraday_charges(turnover, side, *, admin_square_off=False, delivery=False):
    if side not in ("BUY", "SELL") or turnover < 0:
        raise ValueError("Expected nonnegative turnover and BUY or SELL.")
    val = float(turnover)
    brok = min(20.0, val * (0.003 if delivery else 0.0003))
    ex = val * 0.000030699
    sebi = val * 0.000001
    ipft = val * 0.000000001
    admin = 50.0 if admin_square_off else 0.0
    dp = 12.5 if delivery and side == "SELL" else 0.0
    stt = val * 0.001 if delivery else (val * 0.00025 if side == "SELL" else 0.0)
    stamp = val * (0.00015 if delivery else 0.00003) if side == "BUY" else 0.0
    gst = (brok + ex + sebi + ipft + admin + dp) * 0.18
    result = {
        "dp": round(dp, 2),
        "brokerage": round(brok, 2),
        "exchange": round(ex, 2),
        "sebi": round(sebi, 2),
        "ipft": round(ipft, 2),
        "stt": round(stt, 2),
        "stamp": round(stamp, 2),
        "admin": round(admin, 2),
        "gst": round(gst, 2),
    }
    result["fee"] = round(sum(result.values()), 2)
    return result


def fast_intraday_fee(turnover, side, *, delivery=False):
    """Fast scalar fee calculation without allocating dictionaries."""
    val = float(turnover)
    brok = 20.0 if val * 0.0003 > 20.0 else val * (0.003 if delivery else 0.0003)
    ex = val * 0.000030699
    sebi = val * 0.000001
    ipft = val * 0.000000001
    is_sell = side == "SELL"
    stt = val * 0.001 if delivery else (val * 0.00025 if is_sell else 0.0)
    stamp = (val * (0.00015 if delivery else 0.00003)) if not is_sell else 0.0
    gst = (brok + ex + sebi + ipft) * 0.18
    return round(round(brok, 2) + round(stt, 2) + round(ex, 2) + round(sebi, 2) + round(ipft, 2) + round(stamp, 2) + round(gst, 2), 2)


def us_intraday_charges(turnover, side, shares=0):
    """US Regulated Equities (Zero-commission retail broker: SEC 31 + FINRA TAF on sell)."""
    val = float(turnover)
    sec = round(val * 0.0000278, 2) if side == "SELL" else 0.0
    taf = round(min(8.30, float(shares) * 0.000166), 2) if side == "SELL" else 0.0
    fee = round(sec + taf, 2)
    return {
        "brokerage": 0.0, "stt": 0.0, "exchange": 0.0, "sebi": sec,
        "ipft": taf, "stamp": 0.0, "gst": 0.0, "admin": 0.0, "dp": 0.0,
        "fee": fee
    }


def fast_us_fee(turnover, side, shares=0):
    """Fast US regulatory fee (SEC + FINRA on sales)."""
    if side != "SELL":
        return 0.0
    val = float(turnover)
    sec = round(val * 0.0000278, 2)
    taf = round(min(8.30, float(shares) * 0.000166), 2)
    return round(sec + taf, 2)


def affordable_shares(price, budget, cash):
    """Maximum whole shares satisfying notional cap and fee-inclusive cash."""
    low, high = 0, max(0, int(min(budget, cash) / price))
    while low < high:
        quantity = (low + high + 1) // 2
        total = quantity * price + fast_intraday_fee(quantity * price, "BUY")
        if total <= cash:
            low = quantity
        else:
            high = quantity - 1
    return low
