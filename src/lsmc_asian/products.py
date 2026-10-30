"""Option contracts.

Each product exposes

* ``exercise_values(paths)``: (M, N+1) array, the payoff received if the option
  is exercised at grid date j (column 0 is never used for exercise);
* ``payoff(paths)``: the payoff at maturity, i.e. ``exercise_values(paths)[:, -1]``;
* ``state_arrays(paths)``: tuple of (M, N+1) arrays; column j of each is a regressor
  for the continuation value at date j.

For the Asian option the payoff on exercise at t_j uses the average to date,
A_j = mean(S_{t_1}, ..., S_{t_j}), so the option is the American (Bermudan)
average-rate option, one of the examples in Longstaff and Schwartz (2001).
Its value depends on (S_j, A_j, r_j), which are all used as regressors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Paths


def _intrinsic(underlying: np.ndarray, strike: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(underlying - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - underlying, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


def running_average(spot: np.ndarray, average: str = "arithmetic") -> np.ndarray:
    """Average of S_{t_1..t_j} for every j; column 0 (no observation yet) is set to S_0."""
    n = spot.shape[1] - 1
    counts = np.arange(1, n + 1)
    out = np.empty_like(spot)
    out[:, 0] = spot[:, 0]
    if average == "arithmetic":
        out[:, 1:] = np.cumsum(spot[:, 1:], axis=1) / counts
    elif average == "geometric":
        out[:, 1:] = np.exp(np.cumsum(np.log(spot[:, 1:]), axis=1) / counts)
    else:
        raise ValueError("average must be 'arithmetic' or 'geometric'")
    return out


@dataclass(frozen=True)
class AsianOption:
    """Fixed-strike average-price option on the discretely monitored average."""

    strike: float
    kind: str = "call"
    average: str = "arithmetic"

    def exercise_values(self, paths: Paths) -> np.ndarray:
        return _intrinsic(running_average(paths.spot, self.average), self.strike, self.kind)

    def payoff(self, paths: Paths) -> np.ndarray:
        return _intrinsic(running_average(paths.spot, self.average)[:, -1], self.strike, self.kind)

    def state_arrays(self, paths: Paths) -> tuple[np.ndarray, ...]:
        return paths.spot, running_average(paths.spot, self.average), paths.short_rate


@dataclass(frozen=True)
class VanillaOption:
    """Plain option on the spot price (European, or Bermudan when priced with LSMC)."""

    strike: float
    kind: str = "put"

    def exercise_values(self, paths: Paths) -> np.ndarray:
        return _intrinsic(paths.spot, self.strike, self.kind)

    def payoff(self, paths: Paths) -> np.ndarray:
        return _intrinsic(paths.spot[:, -1], self.strike, self.kind)

    def state_arrays(self, paths: Paths) -> tuple[np.ndarray, ...]:
        return paths.spot, paths.short_rate
