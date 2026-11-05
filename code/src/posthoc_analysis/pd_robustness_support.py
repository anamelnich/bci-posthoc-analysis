"""Auditable exploratory analyses; raw files and shared outputs remain immutable.

Neural inference is deliberately gated on unresolved acquisition semantics.
All tests below use subjects as independent units, never trials.
"""


from pathlib import Path


import hashlib


import json


import numpy as np


import pandas as pd


from scipy import stats


from .config import PROJECT_ROOT, EXPECTED_SUBJECTS, get_subject_group


def validate_cohort_table(data, keys, sessions=(1, 5)):
    """Require the configured cohort, stable group labels, keys and sessions."""
    required = set(keys) | {"subject_id", "group", "session_id"}
    if not required.issubset(data):
        raise ValueError(f"Missing columns: {sorted(required - set(data))}")
    if data[list(required)].isna().any().any() or data.duplicated(keys).any():
        raise ValueError("Missing identifiers or duplicated observation keys.")
    if set(data.subject_id) != set(EXPECTED_SUBJECTS):
        raise ValueError("Table does not match the complete configured subject roster.")
    labels = data.group.replace({"experimental": "bci"})
    if not labels.eq(data.subject_id.map(get_subject_group)).all():
        raise ValueError("Group labels disagree with config.")
    for subject, cell in data.groupby("subject_id"):
        if set(cell.session_id) != set(sessions):
            raise ValueError(f"{subject}: unexpected or missing sessions.")
    print(f"Validated {len(data):,} unique rows; {len(EXPECTED_SUBJECTS)} subjects, sessions {sessions}.")


def compare_group_changes(summary, metrics, n_permutations=9999, seed=20261013):
    """Welch test of post-pre changes, Holm family correction and robust checks."""
    from statsmodels.stats.multitest import multipletests
    validate_cohort_table(summary, ["subject_id", "session_id"])
    if not metrics or n_permutations < 999:
        raise ValueError("Specify outcomes and at least 999 permutations.")
    rng = np.random.default_rng(seed)
    rows, changes = [], []
    for metric in metrics:
        wide = summary.pivot(index=["subject_id", "group"], columns="session_id", values=metric)
        if set(wide.columns) != {1, 5} or not np.isfinite(wide.to_numpy()).all():
            raise ValueError(f"{metric}: incomplete/nonfinite paired outcomes.")
        delta = (wide[5] - wide[1]).rename("change").reset_index()
        delta["metric"] = metric
        changes.append(delta)
        a = delta.loc[delta.group.isin(["bci", "experimental"]), "change"].to_numpy()
        b = delta.loc[delta.group == "control", "change"].to_numpy()
        if min(len(a), len(b)) < 3 or min(np.var(a, ddof=1), np.var(b, ddof=1)) <= 0:
            raise ValueError(f"{metric}: insufficient independent observations or zero variance.")
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        df = (va + vb)**2 / (va**2 / (len(a)-1) + vb**2 / (len(b)-1))
        estimate = a.mean() - b.mean()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        half = stats.t.ppf(.975, df) * se
        pooled_sd = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        g = (1-3/(4*(len(a)+len(b))-9))*estimate/pooled_sd
        pooled = np.r_[a,b]
        exceed = 0
        for _ in range(n_permutations):
            z = rng.permutation(pooled)
            x,y = z[:len(a)],z[len(a):]
            tp = (x.mean()-y.mean()) / np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))
            exceed += abs(tp) >= abs(t)
        loo = [np.delete(a,i).mean()-b.mean() for i in range(len(a))]
        loo += [a.mean()-np.delete(b,i).mean() for i in range(len(b))]
        rows.append(dict(metric=metric,n_bci=len(a),n_control=len(b),bci_change=a.mean(),
                         control_change=b.mean(),difference=estimate,ci_low=estimate-half,
                         ci_high=estimate+half,t=t,df=df,p=p,hedges_g_change=g,
                         permutation_p=(exceed+1)/(n_permutations+1),
                         loo_difference_min=min(loo),loo_difference_max=max(loo)))
    tests = pd.DataFrame(rows)
    tests["p_holm"] = multipletests(tests.p, method="holm")[1]
    print(f"Compared {len(metrics)} outcomes, subject-level Group x Session contrast; Holm family size {len(metrics)}.")
    return tests, pd.concat(changes, ignore_index=True)

