"""Least-squares Monte Carlo (Longstaff and Schwartz, 2001) for Bermudan-exercise options.

Algorithm (one set of "training" paths):

1. Simulate paths of (S, r) and the pathwise discount factor D_t = exp(-integral_0^t r).
2. Initialise each path's cash flow with the payoff at T and its stopping date tau = N.
3. For j = N-1, ..., j0: on the in-the-money paths, regress the realised cash
   flow discounted back to t_j, CF * D_tau / D_j, on polynomial basis
   functions of the state (S_j, A_j, r_j). Where the immediate exercise value
   exceeds the fitted continuation value, exercise: set CF = exercise value, tau = j.
4. The in-sample price is mean(CF * D_tau).

The in-sample estimate reuses the paths that fitted the policy, so its bias can
go either way: fitting to the same noise pushes it up, an imperfect exercise
rule pushes it down. Applying the frozen policy to independent paths values a
feasible exercise strategy, so the out-of-sample estimate is biased only
downwards (up to Monte Carlo error it is a lower bound on the true price).
Both are reported; the out-of-sample one is the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np

from .analytic import geometric_asian_price
from .model import MarketModel, Paths
from .montecarlo import Estimate, control_variate_estimate, mean_and_error, pair_means
from .products import AsianOption


def monomial_exponents(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    """All monomials of total degree <= ``degree`` in ``n_vars`` variables, as index tuples."""
    terms: list[tuple[int, ...]] = []
    for d in range(degree + 1):
        terms.extend(combinations_with_replacement(range(n_vars), d))
    return terms


def basis_matrix(x: np.ndarray, terms) -> np.ndarray:
    """Evaluate monomials (index tuples) on the rows of x; the empty tuple is the constant."""
    cols = [np.prod(x[:, list(t)], axis=1) if t else np.ones(x.shape[0]) for t in terms]
    return np.column_stack(cols)


@dataclass
class DateRule:
    """Regression fitted at one exercise date: standardisation of the kept state columns and coefficients."""

    columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    terms: list
    coef: np.ndarray

    def continuation(self, states: np.ndarray) -> np.ndarray:
        z = (states[:, self.columns] - self.mean) / self.scale
        return basis_matrix(z, self.terms) @ self.coef


@dataclass
class ExercisePolicy:
    """Fitted exercise rule: one regression per exercise date (None where there were too few ITM paths)."""

    rules: dict[int, DateRule | None] = field(default_factory=dict)
    first_exercise_step: int = 1


def _states_at(arrays, j: int) -> np.ndarray:
    return np.column_stack([a[:, j] for a in arrays])


def _fit_rule(states: np.ndarray, y: np.ndarray, degree: int) -> DateRule | None:
    scale = states.std(axis=0)
    keep = np.flatnonzero(scale > 1e-12 * (np.abs(states).mean(axis=0) + 1e-300))
    terms = monomial_exponents(len(keep), degree)
    if states.shape[0] < 2 * len(terms):
        return None
    z = (states[:, keep] - states[:, keep].mean(axis=0)) / scale[keep]
    coef, *_ = np.linalg.lstsq(basis_matrix(z, terms), y, rcond=None)
    return DateRule(keep, states[:, keep].mean(axis=0), scale[keep], terms, coef)


def fit_policy(paths: Paths, product, degree: int = 2, first_exercise_step: int = 1):
    """Backward induction on ``paths``. Returns (policy, discounted cash flow per path)."""
    n = paths.n_steps
    ex = product.exercise_values(paths)
    arrays = product.state_arrays(paths)
    disc = paths.discount
    rows = np.arange(paths.n_paths)

    cash = ex[:, n].copy()
    tau = np.full(paths.n_paths, n)
    policy = ExercisePolicy(first_exercise_step=first_exercise_step)
    for j in range(n - 1, first_exercise_step - 1, -1):
        itm = np.flatnonzero(ex[:, j] > 0)
        if itm.size == 0:
            policy.rules[j] = None
            continue
        states = _states_at(arrays, j)[itm]
        y = cash[itm] * disc[itm, tau[itm]] / disc[itm, j]
        rule = _fit_rule(states, y, degree)
        policy.rules[j] = rule
        if rule is None:
            continue
        stop = itm[ex[itm, j] > rule.continuation(states)]
        cash[stop] = ex[stop, j]
        tau[stop] = j
    return policy, cash * disc[rows, tau]


def apply_policy(policy: ExercisePolicy, paths: Paths, product) -> np.ndarray:
    """Discounted cash flow per path from following ``policy`` forward on (independent) ``paths``."""
    n = paths.n_steps
    ex = product.exercise_values(paths)
    arrays = product.state_arrays(paths)
    disc = paths.discount
    out = np.zeros(paths.n_paths)
    alive = np.ones(paths.n_paths, dtype=bool)
    for j in range(policy.first_exercise_step, n):
        rule = policy.rules.get(j)
        if rule is None:
            continue
        cand = np.flatnonzero(alive & (ex[:, j] > 0))
        if cand.size == 0:
            continue
        stop = cand[ex[cand, j] > rule.continuation(_states_at(arrays, j)[cand])]
        out[stop] = ex[stop, j] * disc[stop, j]
        alive[stop] = False
    out[alive] = ex[alive, n] * disc[alive, n]
    return out


def _estimate(samples, paths, product, model, maturity, antithetic, control_variate) -> Estimate:
    """Estimate from per-path discounted cash flows; ``n_samples`` counts simulated paths."""
    y = pair_means(samples, antithetic)
    if control_variate:
        if not isinstance(product, AsianOption):
            raise ValueError("the control variate applies to Asian options")
        geo = AsianOption(product.strike, product.kind, "geometric")
        x = pair_means(paths.discount[:, -1] * geo.payoff(paths), antithetic)
        exact = geometric_asian_price(model, product.strike, maturity, paths.n_steps, product.kind)
        est = control_variate_estimate(y, x, exact)
        return Estimate(est.price, est.std_error, samples.shape[0], est.extra)
    price, se = mean_and_error(y)
    return Estimate(price, se, samples.shape[0])


@dataclass(frozen=True)
class LSMCResult:
    in_sample: Estimate
    out_of_sample: Estimate | None
    european: Estimate  # same training paths, exercise only at T
    policy: ExercisePolicy

    @property
    def early_exercise_premium(self) -> float:
        best = self.out_of_sample if self.out_of_sample is not None else self.in_sample
        return best.price - self.european.price


def price_lsmc(
    model: MarketModel,
    product,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    degree: int = 2,
    antithetic: bool = False,
    n_paths_out_of_sample: int | None = None,
    control_variate: bool = False,
    first_exercise_step: int = 1,
) -> LSMCResult:
    """Price a Bermudan-exercise option (exercise on grid dates first_exercise_step..N) by LSMC.

    The training paths fit the policy; if ``n_paths_out_of_sample`` is given, a
    second, independent set of paths is drawn from the same generator to value
    the frozen policy. ``control_variate`` (Asian options) uses the European
    geometric-average option, whose exact price is known, to reduce variance.
    """
    train = model.simulate(maturity, n_steps, n_paths, rng, antithetic=antithetic)
    policy, cash = fit_policy(train, product, degree, first_exercise_step)
    in_sample = _estimate(cash, train, product, model, maturity, antithetic, control_variate)
    european_cash = train.discount[:, -1] * product.payoff(train)
    european = _estimate(european_cash, train, product, model, maturity, antithetic, control_variate)

    oos = None
    if n_paths_out_of_sample:
        test = model.simulate(maturity, n_steps, n_paths_out_of_sample, rng, antithetic=antithetic)
        oos = _estimate(apply_policy(policy, test, product), test, product, model, maturity, antithetic, control_variate)
    return LSMCResult(in_sample=in_sample, out_of_sample=oos, european=european, policy=policy)
