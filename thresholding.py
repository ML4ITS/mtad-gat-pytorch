"""The threshold of the peaks-over-threshold (POT) method.

The method comes from "Anomaly Detection in Streams with Extreme Value Theory" by Siffer et al.
(KDD 2017). It has these steps:

1. Take an initial threshold t. It is the value at the given level in the sorted train scores.
2. Collect the peaks: the train scores above t, minus t.
3. Fit a generalized Pareto distribution to the peaks. The result is a shape gamma and a scale
   sigma. The fit uses the method of Grimshaw.
4. Give the extreme quantile for the risk q:

       z_q = t + (sigma / gamma) * ((q * n / Nt) ** -gamma - 1)

   n is the number of train scores, and Nt is the number of peaks.

The class SPOT of spot.py does the work. With dynamic=False the class gives the value of the
calibration for all the points, thus this module calls initialize() only. With dynamic=True the
class follows the test scores and moves the threshold, and this module gives the mean value.

Two other ways to fit the distribution give worse results. benchmarks/bench_pot.py shows the
numbers. The maximum likelihood of scipy and the probability weighted moments both give a shape
of 0.25 to 0.89 for the example data, and the method of Grimshaw gives a shape near 0. A large
shape gives a much higher threshold, and the f1 of MSL falls from 0.91 to 0.00.
"""

import io
from contextlib import redirect_stdout

import numpy as np

from spot import SPOT


def pot_threshold(
    init_scores: np.ndarray,
    scores: np.ndarray | None = None,
    q: float = 1e-3,
    level: float = 0.98,
    dynamic: bool = False,
) -> float:
    """
    Give the anomaly threshold for a set of scores.

    :param init_scores: scores of the train set, for the calibration
    :param scores: scores of the test set. The method needs them for dynamic=True only.
    :param q: risk, the part of the values that the method accepts above the threshold
    :param level: probability of the initial threshold t
    :param dynamic: if True, the threshold follows the test scores, and the result is the mean
                    of all the values of the threshold
    :return: the threshold
    """
    # The arrays keep their type. The method of Grimshaw searches a root, thus a cast from
    # float32 to float64 moves the threshold by some tenths of a percent.
    init_scores = np.asarray(init_scores)
    stream = np.empty(0, dtype=init_scores.dtype) if scores is None else np.asarray(scores)

    model = SPOT(q)
    model.fit(init_scores, stream)
    with redirect_stdout(io.StringIO()):  # the class prints the steps of the calibration
        model.initialize(level=level, min_extrema=False)

        if not dynamic:
            # spot.py holds no annotations, thus the type of the attribute is not known.
            return float(model.extreme_quantile)  # type: ignore[bad-argument-type]

        result = model.run(dynamic=True, with_alarm=False)

    return float(np.mean(np.asarray(result["thresholds"], dtype=float)))
