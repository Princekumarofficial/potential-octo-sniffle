"""Long-only minute trading environment and tabular Q-learning policy."""

from bisect import bisect_right
from datetime import timedelta
from enum import IntEnum
from math import isfinite
from random import Random


class Action(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnvironment:
    """Trade at the preceding close of each timestamped one-minute return."""

    def __init__(self, forecasts, returns, target_timestamps, *, fee_per_side=0.00005):
        self.forecasts = tuple(map(float, forecasts))
        self.returns = tuple(map(float, returns))
        self.timestamps = tuple(target_timestamps)
        if not len(self.forecasts) == len(self.returns) == len(self.timestamps) > 0:
            raise ValueError("Forecasts, returns, and target timestamps must be aligned and nonempty.")
        if not all(isfinite(x) for x in self.forecasts) or not all(isfinite(x) and x > -1 for x in self.returns):
            raise ValueError("Expected finite forecasts and returns greater than -100%.")
        if not 0 <= fee_per_side < 1:
            raise ValueError("fee_per_side must be in [0, 1).")
        if any(a >= b for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("Target timestamps must be strictly increasing.")
        self.fee_per_side = fee_per_side
        self.ends = [a.date() != b.date() or b - a != timedelta(minutes=1)
                     for a, b in zip(self.timestamps, self.timestamps[1:])] + [True]
        self.reset()

    def reset(self):
        self.index = 0
        self.long = False
        self.equity = [1.0]
        self.entries = self.exits = self.exposed_bars = 0
        self.fees = 0.0
        return self.observation()

    def observation(self):
        decision_time = self.timestamps[self.index] - timedelta(minutes=1)
        elapsed = decision_time.hour * 60 + decision_time.minute - (9 * 60 + 15)
        time_bucket = min(2, max(0, elapsed // 125))
        return self.forecasts[self.index], self.long, time_bucket

    @property
    def actions(self):
        return (Action.HOLD, Action.SELL if self.long else Action.BUY)

    def step(self, action):
        if self.index == len(self.returns):
            raise RuntimeError("Episode is complete. Call reset() before stepping again.")
        if action not in self.actions:
            raise ValueError(f"Action {action} is unavailable for the current position.")

        previous = value = self.equity[-1]
        if action != Action.HOLD:
            fee = value * self.fee_per_side
            self.fees += fee
            value -= fee
            self.long = action == Action.BUY
            self.entries += int(self.long)
            self.exits += int(not self.long)
        if self.long:
            self.exposed_bars += 1
            value *= 1 + self.returns[self.index]
            if self.ends[self.index]:
                fee = value * self.fee_per_side
                self.fees += fee
                value -= fee
                self.long = False
                self.exits += 1
        terminal = self.ends[self.index]
        self.equity.append(value)
        self.index += 1
        next_observation = self.observation() if self.index < len(self.returns) else None
        return next_observation, value / previous - 1, terminal

    def summary(self):
        peak = 1.0
        drawdown = 0.0
        for value in self.equity:
            peak = max(peak, value)
            drawdown = min(drawdown, value / peak - 1)
        return {
            "net_return_pct": 100 * (self.equity[-1] - 1),
            "fees_per_10000_initial_capital": 10_000 * self.fees,
            "entries": self.entries,
            "exits": self.exits,
            "exposure_pct": 100 * self.exposed_bars / len(self.returns),
            "max_drawdown_pct": 100 * drawdown,
        }


class QLearningAgent:
    """30 states: five forecast buckets, two positions, three time buckets."""

    def __init__(self, forecast_edges, *, learning_rate=0.1, discount=0.95, seed=42):
        self.edges = tuple(forecast_edges)
        if len(self.edges) != 4 or not all(isfinite(x) for x in self.edges) or tuple(sorted(self.edges)) != self.edges:
            raise ValueError("Provide four sorted, finite forecast boundaries.")
        self.learning_rate = learning_rate
        self.discount = discount
        self.rng = Random(seed)
        self.q = [[0.0] * 3 for _ in range(30)]

    def state(self, observation):
        forecast, long, time_bucket = observation
        return bisect_right(self.edges, forecast) * 6 + int(long) * 3 + time_bucket

    def choose(self, state, actions, epsilon=0.0):
        if epsilon and self.rng.random() < epsilon:
            return self.rng.choice(actions)
        return max(actions, key=lambda action: self.q[state][action])

    def run(self, environment, *, learn=False, epsilon=0.0):
        observation = environment.reset()
        while observation is not None:
            state = self.state(observation)
            action = self.choose(state, environment.actions, epsilon if learn else 0.0)
            observation, reward, terminal = environment.step(action)
            if learn:
                future = 0.0 if terminal else max(
                    self.q[self.state(observation)][action] for action in environment.actions
                )
                self.q[state][action] += self.learning_rate * (
                    reward + self.discount * future - self.q[state][action]
                )
        return environment.summary()


def forecast_batches(model, loader, device):
    """Collect aligned predictions and targets from a chronological loader."""
    import torch

    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for inputs, actual in loader:
            predictions.append(model(inputs.to(device)).cpu())
            targets.append(actual)
    predicted, actual = torch.cat(predictions), torch.cat(targets)
    if predicted.shape != actual.shape or not torch.isfinite(predicted).all():
        raise ValueError("Expected finite predictions aligned one-to-one with targets.")
    return predicted, actual
