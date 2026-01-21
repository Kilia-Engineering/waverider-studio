import numpy as np
from pysagas.cfd import OPM


def test_pm():
    """Test Prandtl-Meyer solver."""
    M1 = 1.5
    p1 = 101e3
    T1 = 288
    theta = np.deg2rad(15)
    M2, p2, T2 = OPM._solve_pm(theta, M1, p1, T1)

    assert round(M2) == 2.0, f"incorrect M2 (M2={M2})"
    assert round(p2) == 45998, f"incorrect p2 (p2={p2})"
    assert round(T2) == 230, f"incorrect T2 (T2={T2})"
