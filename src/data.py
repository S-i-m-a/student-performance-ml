"""SYNTHETIC dataset generator. This is NOT real student data.
Pass/Fail is defined from a hidden 'final_score' (>=40 -> Pass), and final_score
is NOT included as a feature (prevents target leakage)."""
import numpy as np, pandas as pd
from .config import DATA_PATH, SEED, FEATURES, TARGET

def generate(n=1000, seed=SEED):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "study_hours": r.uniform(0, 10, n).round(1),
        "attendance": r.uniform(40, 100, n).round(1),
        "previous_score": r.normal(60, 18, n).clip(0, 100).round(1),
        "assignment_completion": r.uniform(0, 100, n).round(1),
        "sleep_hours": r.normal(7, 1.2, n).clip(3, 10).round(1),
    })
    final = (0.30*d.previous_score + 0.20*d.attendance + 0.15*d.assignment_completion
             + 3.0*d.study_hours + 1.0*d.sleep_hours + r.normal(0, 6, n))
    d[TARGET] = (final >= 62).astype(int)
    # inject ~3% missing values to demonstrate imputation
    for c in ["attendance", "sleep_hours"]:
        d.loc[r.choice(n, int(0.03*n), replace=False), c] = np.nan
    return d

def load():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)
