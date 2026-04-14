"""Statistics for offline and online posthoc analysis outputs."""

from __future__ import annotations

import numpy as np

from .modeling import cross_validated_side_decoding


def permutation_test_multiclass_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    n_perm: int = 500,
    random_state: int = 42,
) -> dict[str, float | np.ndarray]:
    """Permutation test for three-class CV decoding accuracy."""

    rng = np.random.default_rng(random_state)
    observed = cross_validated_side_decoding(X, y)["accuracy"]
    null = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = cross_validated_side_decoding(X, rng.permutation(y))["accuracy"]
    p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return {"observed_accuracy": float(observed), "null_distribution": null, "p_value": float(p)}


def reaction_time_from_triggers(samples: np.ndarray, codes: np.ndarray, sfreq: float) -> np.ndarray:
    """Compute per-trial RT (ms) from condition->response trigger pairs.

    Expected code triplets: fixation(4), condition(8/32/44), response(64).
    """

    rt_ms = []
    i = 0
    while i + 2 < len(codes):
        if codes[i] == 4 and codes[i + 1] in (8, 32, 44) and codes[i + 2] == 64:
            rt_ms.append((samples[i + 2] - samples[i + 1]) * 1000.0 / sfreq)
            i += 3
        else:
            i += 1
    return np.asarray(rt_ms, dtype=float)
