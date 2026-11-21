import numpy as np
import pandas as pd

DEFAULT_COST = 0.0005

def apply_costs(strategy_ret, signal):
    turnover = signal.diff().abs().fillna(0.0)
    return strategy_ret - turnover * DEFAULT_COST

def finalize_strategy(ret, signal, cost):
    gross = signal.shift(fill_value=0.0) * ret.fillna(0.0)
    return gross - signal.diff().fillna(signal).abs() * cost

def to_daily_returns(minute_ret):
    minute_ret = minute_ret.dropna()
    log_ret = np.log1p(minute_ret)
    daily_log = log_ret.groupby(log_ret.index.date).sum()
    daily = np.expm1(daily_log)
    daily.index = pd.to_datetime(daily.index)
    return daily

def sharpe(daily):
    deviation = daily.std()
    if not np.isfinite(deviation) or deviation == 0:
        return None
    return float(daily.mean() / deviation * np.sqrt(252))

def sortino(daily):
    downside = np.sqrt(daily.clip(upper=0).pow(2).mean())
    if not np.isfinite(downside) or downside == 0:
        return None
    return float(daily.mean() / downside * np.sqrt(252))

def max_drawdown(minute_ret):
    eq = (1 + minute_ret.fillna(0)).cumprod()
    peak = eq.cummax().clip(lower=1.0)
    return float((eq / peak - 1).min())


class StrategyProfitCalculator:
    """Summarize one flat-start backtest with P&L attributed to starting capital."""

    def __init__(self, close, signal, net_returns):
        self.market = close.pct_change().fillna(0.0)
        self.position = signal.shift(fill_value=0.0)
        self.signal = signal
        self.net = net_returns
        self.equity = (1 + self.net).cumprod()

    def summary(self, *, include_curves=False):
        daily = to_daily_returns(self.net)
        market_daily = to_daily_returns(self.market)
        gross = self.position * self.market
        opening_equity = self.equity.shift(fill_value=1.0)
        gross_pnl = opening_equity * gross
        active_moves = gross[(self.position != 0) & (self.market != 0)]
        worst_day = daily.idxmin()
        report = {
            "start": self.net.index[0].isoformat(),
            "end": self.net.index[-1].isoformat(),
            "sessions": len(daily),
            "cumulative_return": float(self.equity.iloc[-1] - 1),
            "buy_and_hold_return": float((1 + self.market).prod() - 1),
            "market_daily_volatility": float(market_daily.std()) if len(market_daily) > 1 else None,
            "sharpe": sharpe(daily),
            "sortino": sortino(daily),
            "dd": max_drawdown(self.net),
            "position_changes": int(self.signal.diff().fillna(self.signal).ne(0).sum()),
            "long_exposure": float(self.position.gt(0).mean()),
            "short_exposure": float(self.position.lt(0).mean()),
            "long_pnl": float(gross_pnl[self.position > 0].sum()),
            "short_pnl": float(gross_pnl[self.position < 0].sum()),
            "cost_drag": float((opening_equity * (gross - self.net)).sum()),
            "directional_hit_rate": float(active_moves.gt(0).mean()) if len(active_moves) else None,
            "worst_day": worst_day.date().isoformat(),
            "worst_day_return": float(daily.loc[worst_day]),
        }
        report.update(self.net.attrs.get("execution", {}))
        if include_curves:
            report["daily_curves"] = pd.DataFrame({
                "strategy": (1 + daily).cumprod() - 1,
                "benchmark": (1 + market_daily).cumprod() - 1,
            })
        return report


def calmar_ratio(returns, drawdown):
    years = (returns.index[-1] - returns.index[0]).total_seconds() / (365.25 * 86400)
    growth = float((1 + returns.fillna(0)).prod())
    if years <= 0 or growth <= 0 or drawdown >= 0:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        score = np.expm1(np.log(growth) / years) / abs(drawdown)
    return float(score) if np.isfinite(score) else None
