"""Compare three ways to fit the distribution of the POT method.

The POT method fits a generalized Pareto distribution to the peaks of the train scores. The
shape and the scale of that distribution give the threshold. This benchmark compares:

- grimshaw:  the method of the class SPOT in spot.py, which the project uses
- scipy mle: the maximum likelihood of scipy.stats.genpareto
- pwm:       the probability weighted moments of Hosking and Wallis

The benchmark uses the example results in `output/`. Run it with:

    uv run benchmarks/bench_pot.py
"""

import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_methods import adjust_predicts, calc_point2point  # noqa: E402
from spot import SPOT  # noqa: E402

# The values of q and of the level come from predict.py.
CASES = [
    ("SMD 1-1", "output/SMD/1-1/27062021_114402", 0.001, 0.9950),
    ("MSL", "output/MSL/27062021_111641", 0.001, 0.90),
    ("SMAP", "output/SMAP/27062021_112545", 0.005, 0.90),
]


def peaks_of(scores, level):
    """Give the initial threshold and the peaks above it."""
    level = level - np.floor(level)
    t = np.sort(scores)[int(level * scores.size)]
    return t, scores[scores > t] - t


def extreme_quantile(t, gamma, sigma, r):
    """The formula of Siffer et al. for the quantile of the risk."""
    if abs(gamma) < 1e-10:  # the distribution is an exponential distribution
        return float(t - sigma * np.log(r))
    return float(t + (sigma / gamma) * (r**-gamma - 1))


def fit_grimshaw(scores, peaks, q, level):
    model = SPOT(q)
    model.fit(scores, np.empty(0, dtype=scores.dtype))
    with redirect_stdout(io.StringIO()):
        model.initialize(level=level, min_extrema=False)
    gamma, sigma, _ = model._grimshaw()
    return gamma, sigma


def fit_scipy(scores, peaks, q, level):
    gamma, _, sigma = genpareto.fit(peaks, floc=0)
    return gamma, sigma


def fit_pwm(scores, peaks, q, level):
    """Probability weighted moments. The shape of scipy is the negative shape of Hosking."""
    x = np.sort(peaks)
    p = (np.arange(x.size) + 0.35) / x.size
    a0 = x.mean()
    a1 = np.mean(x * (1 - p))
    k = a0 / (a0 - 2 * a1) - 2
    sigma = 2 * a0 * a1 / (a0 - 2 * a1)
    return -k, sigma


FITS = [("grimshaw", fit_grimshaw), ("scipy mle", fit_scipy), ("pwm", fit_pwm)]


def main():
    header = f"{'dataset':<9} {'fit':<10} {'gamma':>9} {'sigma':>9} {'threshold':>11} {'f1':>9} {'precision':>10} {'recall':>8} {'ms':>7}"
    print(header)
    print("-" * len(header))

    for name, path, q, level in CASES:
        train = pl.read_parquet(f"{path}/train_output.parquet")["A_Score_Global"].to_numpy()
        test_frame = pl.read_parquet(f"{path}/test_output.parquet")
        test = test_frame["A_Score_Global"].to_numpy()
        labels = test_frame["A_True_Global"].to_numpy()

        t, peaks = peaks_of(train, level)
        r = q * train.size / peaks.size

        for label, fit in FITS:
            start = time.perf_counter()
            gamma, sigma = fit(train, peaks, q, level)
            threshold = extreme_quantile(t, gamma, sigma, r)
            milliseconds = (time.perf_counter() - start) * 1000

            pred, _ = adjust_predicts(test, labels, threshold, calc_latency=True)
            f1, precision, recall, *_ = calc_point2point(pred, labels)

            print(
                f"{name:<9} {label:<10} {gamma:>9.5f} {sigma:>9.5f} {threshold:>11.6f} "
                f"{f1:>9.6f} {precision:>10.6f} {recall:>8.6f} {milliseconds:>7.1f}"
            )
        print(f"{'':<9} peaks={peaks.size} of {train.size} scores, initial threshold t={t:.6f}")
        print()


if __name__ == "__main__":
    main()
