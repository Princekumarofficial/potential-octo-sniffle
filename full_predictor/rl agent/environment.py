"""Shared-cash, long-only research environment. Actions: HOLD=0 BUY=1 SELL=2."""
import numpy as np
from GA_STRATEGY.engine.fees import intraday_charges, us_intraday_charges
from markets import MARKETS, local_index
from risk_actions import ACTION_RISK, BUY_ACTIONS, ACTION_COUNT


class TradingEnv:
    def __init__(self, times, prices, features, capital=1_000_000., allocation=.04, *, market="india", source_timezone=None, availability=None, max_positions=None, drawdown_penalty=0., risk_actions=False):
        if capital <= 0 or not 0 < allocation <= 1 or len(times) < 2:
            raise ValueError("Invalid account or empty episode")
        if not np.isfinite(prices).all() or (prices <= 0).any() or not np.isfinite(features).all():
            raise ValueError("Nonfinite inputs or nonpositive prices")
        self.risk_actions = risk_actions
        self.market = market
        self.settings = MARKETS[market]
        times = local_index(times, market, source_timezone)
        minutes = times.hour * 60 + times.minute
        if ((minutes < self.settings["start"]) | (minutes > self.settings["end"])).any():
            raise ValueError("Environment requires regular-session bars in exchange-local time")
        self.times, self.prices, self.features = times, prices, features
        self.capital, self.allocation = capital, allocation
        self.n = prices.shape[1]
        self.available = np.ones(prices.shape[:2], bool) if availability is None else np.asarray(availability, bool)
        self.max_positions = self.n if max_positions is None else max_positions
        self.drawdown_penalty = drawdown_penalty
        if self.max_positions < 1 or drawdown_penalty < 0:
            raise ValueError("Invalid position cap or drawdown penalty")
        self.session_last = np.zeros_like(self.available)
        dates = times.normalize()
        for date in dates.unique():
            indices = np.flatnonzero(dates == date)
            for stock in range(self.n):
                observed = indices[self.available[indices, stock]]
                if len(observed):
                    self.session_last[observed[-1], stock] = True
        self.reset()

    def reset(self):
        self.i, self.cash, self.equity = 0, self.capital, self.capital
        self.peak, self.drawdown = self.capital, 0.
        self.entry = np.zeros(self.n)
        self.sl = np.zeros(self.n)
        self.tp = np.zeros(self.n)
        self.quantity = np.zeros(self.n, dtype=int)
        self.fills, self.curve = [], [self.capital]
        return self.observation()

    def observation(self):
        close = self.prices[self.i, :, 1]
        stamp = self.times[self.i]
        account = np.column_stack((self.quantity > 0, self.quantity * close / self.equity,
                                   np.full(self.n, self.cash / self.equity),
                                   np.full(self.n, (stamp.hour * 60 + stamp.minute) / 1440)))
        if self.risk_actions:
            pnl = np.divide(close, self.entry, out=np.ones(self.n), where=self.entry > 0) - 1
            account = np.column_stack((account, pnl, self.sl, self.tp))
        return np.concatenate((self.features[self.i], account), axis=1).astype(np.float32)

    def mask(self):
        mask = np.column_stack((np.ones(self.n, bool), (self.quantity == 0) & self.available[self.i], (self.quantity > 0) & self.available[self.i]))
        if self.risk_actions:
            mask = np.column_stack((mask, np.repeat(mask[:, 1:2], 8, axis=1)))
        return mask

    def charges(self, price, qty, side):
        if self.market == "us":
            return us_intraday_charges(float(qty * price), side, int(qty))["fee"]
        return intraday_charges(float(qty * price), side)["fee"]

    def trade(self, stock, price, side, timestamp, budget=None, risk=None, reason="agent"):
        qty = self.quantity[stock] if side == "SELL" else max(0, int(min(budget, self.cash) / price))
        if side == "BUY":
            while qty and qty * price + self.charges(price, qty, side) > self.cash:
                qty -= 1
        if not qty:
            return
        fee = self.charges(price, qty, side)
        direction = 1 if side == "BUY" else -1
        self.cash -= direction * qty * price + fee
        self.quantity[stock] += direction * qty
        if side == "BUY":
            self.entry[stock] = price
            self.sl[stock], self.tp[stock] = risk or (0., 0.)
        self.fills.append(dict(timestamp=str(timestamp), stock_index=stock, side=side,
                               shares=int(qty), price=float(price), fee=fee, reason=reason,
                               stop_loss=float(self.sl[stock]), take_profit=float(self.tp[stock]),
                               entry_price=float(self.entry[stock])))
        if side == "SELL":
            self.entry[stock] = self.sl[stock] = self.tp[stock] = 0.

    def step(self, actions, priorities=None):
        actions = np.asarray(actions)
        if actions.shape != (self.n,) or not np.isin(actions, range(ACTION_COUNT) if self.risk_actions else [0, 1, 2]).all():
            raise ValueError("One HOLD/BUY/SELL action required per stock")
        before = self.equity
        self.i += 1
        timestamp = self.times[self.i]
        opens, closes = self.prices[self.i, :, 0], self.prices[self.i, :, 1]
        same_day = timestamp.date() == self.times[self.i-1].date()
        minute = timestamp.hour * 60 + timestamp.minute
        stopped = set()
        if self.risk_actions:
            for j in range(self.n):
                if self.quantity[j] and self.available[self.i, j]:
                    change = opens[j] / self.entry[j] - 1
                    if change <= -self.sl[j] or change >= self.tp[j]:
                        self.trade(j, opens[j], "SELL", timestamp, reason="stop_loss" if change <= -self.sl[j] else "take_profit")
                        stopped.add(j)
        # Actions from yesterday expire. Sales release cash before purchases.
        if same_day:
            for j in range(self.n):
                if actions[j] == 2 and self.quantity[j] and self.available[self.i, j]:
                    self.trade(j, opens[j], "SELL", timestamp)
            budget = (self.cash + np.dot(self.quantity, opens)) * self.allocation
            order = range(self.n) if priorities is None else np.argsort(-np.asarray(priorities), kind="stable")
            for j in order:
                if np.count_nonzero(self.quantity) >= self.max_positions:
                    break
                if self.available[self.i, j] and self.available[self.i-1, j] and actions[j] in (BUY_ACTIONS if self.risk_actions else [1]) and j not in stopped and not self.quantity[j] and minute < self.settings["entry"]:
                    self.trade(j, opens[j], "BUY", timestamp, budget, risk=ACTION_RISK[int(actions[j])] if self.risk_actions else None)
        if self.risk_actions:
            for j in range(self.n):
                if self.quantity[j] and self.available[self.i, j]:
                    change = closes[j] / self.entry[j] - 1
                    if change <= -self.sl[j] or change >= self.tp[j]:
                        self.trade(j, closes[j], "SELL", timestamp, reason="stop_loss" if change <= -self.sl[j] else "take_profit")
        done = self.i == len(self.times)-1
        last_day_bar = done or self.times[self.i+1].date() != timestamp.date()
        for j in range(self.n):
            if self.available[self.i, j] and self.quantity[j] and (minute >= self.settings["exit"] or last_day_bar or self.session_last[self.i, j]):
                self.trade(j, closes[j], "SELL", timestamp, reason="session_end")
        self.equity = float(self.cash + np.dot(self.quantity, closes))
        self.curve.append(self.equity)
        self.peak = max(self.peak, self.equity)
        drawdown = 1 - self.equity / self.peak
        reward = float(np.log(self.equity / before) - self.drawdown_penalty * max(0., drawdown - self.drawdown))
        self.drawdown = drawdown
        return self.observation(), reward, done, {"equity": self.equity}
