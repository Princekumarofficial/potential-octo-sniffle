"""Explicit exchange-local clock conversion; never infer market from prices."""
import pandas as pd

MARKETS = {
    "india": dict(timezone="Asia/Kolkata", currency="INR", start=555, end=930, entry=900, exit=915),
    "us": dict(timezone="America/New_York", currency="USD", start=570, end=960, entry=945, exit=955),
}


def local_index(index, market, source_timezone=None):
    settings = MARKETS[market]
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        index = index.tz_localize(source_timezone or settings["timezone"], ambiguous="raise", nonexistent="raise")
    # Convert with DST rules before stripping timezone for existing strategy functions.
    return index.tz_convert(settings["timezone"]).tz_localize(None)
