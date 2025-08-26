from __future__ import annotations
import numpy as np
import pandas as pd

def generate_synthetic(n_normal: int = 2000, n_anom: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_normal)
    s1 = 0.02 * t + 2 * np.sin(t / 25) + rng.normal(0, 0.3, size=n_normal)
    s2 = 0.01 * t + 0.7 * np.cos(t / 33) + rng.normal(0, 0.25, size=n_normal)
    df = pd.DataFrame({'time': t, 'sensor_1': s1, 'sensor_2': s2})
    idx = rng.choice(n_normal, size=n_anom, replace=False)
    df.loc[idx, 'sensor_1'] += rng.normal(5, 1.0, size=n_anom)
    df.loc[idx, 'sensor_2'] -= rng.normal(4, 1.2, size=n_anom)
    return df
