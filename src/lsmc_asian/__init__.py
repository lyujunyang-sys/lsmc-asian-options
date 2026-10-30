"""Least-squares Monte Carlo pricing of Asian options under Vasicek stochastic interest rates."""

__version__ = "1.0.0"

from .analytic import (  # noqa: E402
    black_scholes,
    discounted_average_forward,
    european_price,
    geometric_asian_price,
    zero_coupon_bond,
)
from .lsmc import price_lsmc  # noqa: E402
from .model import MarketModel, Paths  # noqa: E402
from .montecarlo import Estimate, price_european  # noqa: E402
from .products import AsianOption, VanillaOption  # noqa: E402

__all__ = [
    "AsianOption",
    "Estimate",
    "MarketModel",
    "Paths",
    "VanillaOption",
    "black_scholes",
    "discounted_average_forward",
    "european_price",
    "geometric_asian_price",
    "price_european",
    "price_lsmc",
    "zero_coupon_bond",
]
