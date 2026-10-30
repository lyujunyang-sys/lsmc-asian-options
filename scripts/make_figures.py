"""Draw the figures in results/figures/ from the committed results (and data/raw/ if downloaded).

    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lsmc_asian import AsianOption, MarketModel  # noqa: E402
from lsmc_asian.calibration import rolling_volatility  # noqa: E402
from lsmc_asian.data import load_rate, load_sp500  # noqa: E402
from lsmc_asian.experiments import load_config, rng_for  # noqa: E402
from lsmc_asian.lsmc import fit_policy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def baseline_model():
    cfg = load_config(ROOT / "configs" / "appendix_baseline.json")
    return cfg, MarketModel(**cfg["market"])


def fig_paths():
    cfg, m = baseline_model()
    T, N = cfg["contract"]["maturity"], cfg["contract"]["n_steps"]
    p = m.simulate(T, N, 2000, rng_for(cfg["seed"], "figure-paths"))
    t = p.times
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i in range(20):
        a1.plot(t, p.spot[i], color=INK2, alpha=0.25, lw=1)
    avg = np.cumsum(p.spot[0, 1:]) / np.arange(1, N + 1)
    a1.plot(t, p.spot[0], color=BLUE, label="one path: spot $S_t$")
    a1.plot(t[1:], avg, color=ORANGE, label="its running average $A_t$")
    a1.axhline(cfg["contract"]["strike"], color=INK, lw=1, ls="--", label="strike K")
    a1.set(title="Asset paths", xlabel="time (years)", ylabel="price")
    a1.legend(loc="upper left")

    for i in range(20):
        a2.plot(t, 100 * p.short_rate[i], color=INK2, alpha=0.25, lw=1)
    mean = m.theta + (m.r0 - m.theta) * np.exp(-m.kappa * t)
    sd = np.sqrt(m.sigma_r**2 / (2 * m.kappa) * (1 - np.exp(-2 * m.kappa * t)))
    a2.fill_between(t, 100 * (mean - 2 * sd), 100 * (mean + 2 * sd), color=BLUE, alpha=0.12, lw=0, label="exact mean ± 2 sd")
    a2.plot(t, 100 * mean, color=BLUE, label="exact mean")
    a2.plot(t, 100 * p.short_rate[:, :].mean(axis=0), color=ORANGE, lw=1.5, ls=":", label="simulated mean (2,000 paths)")
    a2.set(title="Vasicek short rate", xlabel="time (years)", ylabel="rate (%)")
    a2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "sample_paths.png", dpi=150)
    plt.close(fig)


def fig_convergence():
    df = pd.read_csv(ROOT / "results" / "convergence" / "convergence.csv")
    paths = df[df.study == "paths"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for method, color in [("MC", INK2), ("MC + antithetic + control variate", BLUE)]:
        d = paths[paths.method == method]
        a1.loglog(d.n_paths, d.std_error, "o-", color=color, ms=5, label=method)
    d = paths[paths.method == "MC"]
    ref = d.std_error.iloc[0] * np.sqrt(d.n_paths.iloc[0] / d.n_paths)
    a1.loglog(d.n_paths, ref, color=INK2, lw=1, ls="--", label=r"$\propto 1/\sqrt{M}$")
    a1.set(title="Standard error, European-exercise Asian call", xlabel="paths M", ylabel="standard error")
    a1.legend(loc="lower left")

    for method, color in [("LSMC in-sample", ORANGE), ("LSMC out-of-sample", BLUE)]:
        d = paths[paths.method == method]
        a2.fill_between(d.n_paths, d.price - 1.96 * d.std_error, d.price + 1.96 * d.std_error, color=color, alpha=0.12, lw=0)
        a2.semilogx(d.n_paths, d.price, "o-", color=color, ms=5, label=f"{method} (95% CI)")
    a2.set(title="LSMC price, Bermudan-exercise Asian call", xlabel="paths M (each set)", ylabel="price")
    a2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)
    plt.close(fig)


def fig_exercise():
    cfg, m = baseline_model()
    T, N = cfg["contract"]["maturity"], cfg["contract"]["n_steps"]
    product = AsianOption(cfg["contract"]["strike"], "call")
    p = m.simulate(T, N, 20000, rng_for(cfg["seed"], "figure-exercise"))
    policy, _ = fit_policy(p, product)
    # replay the fitted rule on its own paths to find each path's stopping date
    ex = product.exercise_values(p)
    arrays = product.state_arrays(p)
    alive = np.ones(p.n_paths, bool)
    stops = np.zeros(N + 1)
    for j in range(1, N):
        rule = policy.rules.get(j)
        if rule is None:
            continue
        cand = np.flatnonzero(alive & (ex[:, j] > 0))
        if cand.size:
            states = np.column_stack([a[cand, j] for a in arrays])
            s = cand[ex[cand, j] > rule.continuation(states)]
            stops[j] = s.size
            alive[s] = False
    stops[N] = np.sum(alive & (ex[:, N] > 0))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    t = p.times
    ax.bar(t[1:N], 100 * stops[1:N] / p.n_paths, width=0.8 * T / N, color=BLUE, label="early exercise")
    ax.bar(t[N], 100 * stops[N] / p.n_paths, width=0.8 * T / N, color=ORANGE, label="held to maturity, in the money")
    ax.set(title="When the fitted LSMC rule exercises (Bermudan Asian call)", xlabel="exercise date (years)", ylabel="share of paths (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "exercise_times.png", dpi=150)
    plt.close(fig)


def fig_sensitivity():
    df = pd.read_csv(ROOT / "results" / "sensitivity" / "sensitivity.csv")
    cfg = load_config(ROOT / "configs" / "sensitivity.json")
    base_values = {**cfg["market"], "strike": cfg["contract"]["strike"]}
    panels = [("sigma_r", r"rate volatility $\sigma_r$"), ("strike", "strike K")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (param, label) in zip(axes, panels):
        d = pd.concat([df[df.parameter == param], df[df.parameter == "base"]]).copy()
        d["x"] = d["value"].astype(float).fillna(base_values[param])
        d = d.sort_values("x")
        for col, se, color, name in [("asian_bermudan_oos", "asian_bermudan_oos_se", ORANGE, "arithmetic, Bermudan (LSMC, 95% CI)"),
                                     ("asian_european", "asian_european_se", BLUE, "arithmetic, European (MC + CV, 95% CI)"),
                                     ("geometric_exact", None, AQUA, "geometric, European (exact)")]:
            yerr = 1.96 * d[se] if se else None
            ax.errorbar(d.x, d[col], yerr=yerr, fmt="o-", color=color, ms=5, capsize=3, label=name)
        ax.set(title=f"Asian call price vs {label}", xlabel=label, ylabel="price")
    axes[0].legend(loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity.png", dpi=150)
    plt.close(fig)


def fig_data():
    raw = ROOT / "data" / "raw"
    if not (raw / "sp500_close.csv").exists():
        print("data/raw/ not found: skipping market_data.png (run scripts/download_data.py)")
        return
    px = load_sp500(raw / "sp500_close.csv")
    vol = rolling_volatility(px)
    r10 = load_rate(raw / "fred_DGS10.csv")
    r3m = load_rate(raw / "fred_DGS3MO.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(px.index, px, color=BLUE, lw=1.2)
    axes[0].set(title="S&P 500 close", ylabel="index level")
    axes[1].plot(vol.index, 100 * vol, color=BLUE, lw=1.2)
    axes[1].set(title="Rolling 252-day volatility", ylabel="annualised (%)")
    axes[2].plot(r10.index, 100 * r10, color=BLUE, lw=1.2, label="10-year (used)")
    axes[2].plot(r3m.index, 100 * r3m, color=INK2, lw=1.2, label="3-month")
    axes[2].set(title="US Treasury yields", ylabel="yield (%)")
    axes[2].legend(loc="upper left")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=0)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "market_data.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_paths()
    fig_convergence()
    fig_exercise()
    fig_sensitivity()
    fig_data()
    print(f"figures written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
