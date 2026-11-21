"""Current FYERS Standard NSE equity intraday tariff, checked 2026-11-16.

Sources: https://fyers.in/charges-list and https://fyers.in/pricing
Amounts are calculated per executed order and rounded to paise. Contract-note
aggregation/rounding may differ. These are current-rate scenario estimates.
"""
from decimal import Decimal, ROUND_HALF_UP

COMPONENTS = ("brokerage", "stt", "exchange", "sebi", "ipft", "stamp", "gst", "admin", "dp")


def intraday_charges(turnover, side, *, admin_square_off=False, delivery=False):
    if side not in ("BUY", "SELL") or turnover < 0:
        raise ValueError("Expected nonnegative turnover and BUY or SELL.")
    value = Decimal(str(turnover))
    brokerage = min(Decimal("20"), value * Decimal("0.003" if delivery else "0.0003"))
    exchange = value * Decimal("0.000030699")
    sebi = value * Decimal("0.000001")
    ipft = value * Decimal("0.000000001")
    admin = Decimal("50") if admin_square_off else Decimal("0")
    dp = Decimal("12.5") if delivery and side == "SELL" else Decimal("0")
    raw = dict(dp=dp,
        brokerage=brokerage, exchange=exchange, sebi=sebi, ipft=ipft,
        stt=value * Decimal("0.001") if delivery else (value * Decimal("0.00025") if side == "SELL" else Decimal("0")),
        stamp=value * Decimal("0.00015" if delivery else "0.00003") if side == "BUY" else Decimal("0"),
        admin=admin,
        gst=(brokerage + exchange + sebi + ipft + admin + dp) * Decimal("0.18"),
    )
    result = {key: float(amount.quantize(Decimal(".01"), rounding=ROUND_HALF_UP))
              for key, amount in raw.items()}
    result["fee"] = round(sum(result.values()), 2)
    return result


def affordable_shares(price, budget, cash):
    """Maximum whole shares satisfying notional cap and fee-inclusive cash."""
    low, high = 0, max(0, int(min(budget, cash) / price))
    while low < high:
        quantity = (low + high + 1) // 2
        total = quantity * price + intraday_charges(quantity * price, "BUY")["fee"]
        if total <= cash:
            low = quantity
        else:
            high = quantity - 1
    return low
