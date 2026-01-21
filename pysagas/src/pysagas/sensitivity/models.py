import numpy as np
from pysagas.geometry import Cell


def piston_sensitivity(cell: Cell, p_i: int, **kwargs):
    """Calculates the pressure-parameter sensitivity using
    local piston theory.

    Parameters
    ----------
    cell : Cell
        The cell object.

    p_i : int
        The index of the parameter to find the sensitivity for. This is used to
        index cell.dndp.
    """
    M_l = cell.flowstate.M
    if M_l < 1.0:
        # Subsonic cell, skip
        return 0

    dPdp = (
        cell.flowstate.rho
        * cell.flowstate.a
        * np.dot(cell.flowstate.vec, -cell.dndp[:, p_i])
    )
    return dPdp


def van_dyke_sensitivity(
    cell: Cell,
    p_i,
    **kwargs,
):
    """
    Calculates the pressure-parameter sensitivity using
    Van Dyke second-order theory.

     Parameters
    ----------
    cell : Cell
        The cell object.

    p_i : int
        The index of the parameter to find the sensitivity for. This is used to
        index cell.dndp.
    """
    M_l = cell.flowstate.M
    if M_l < 1.0:
        # Subsonic cell, skip
        return 0

    piston = piston_sensitivity(cell=cell, p_i=p_i)
    dPdp = piston * M_l / (M_l**2 - 1) ** 0.5
    return dPdp


def isentropic_sensitivity(cell: Cell, p_i: int, **kwargs):
    """Calculates the pressure-parameter sensitivity using
    the isentropic flow relation directly."""
    gamma = cell.flowstate.gamma
    power = (gamma + 1) / (gamma - 1)
    dPdW = (cell.flowstate.P * gamma / cell.flowstate.a) * (
        1 + cell.flowstate.v * (gamma - 1) / (2 * cell.flowstate.a)
    ) ** power
    dWdn = -cell.flowstate.vec
    dndp = cell.dndp[:, p_i]
    dPdp = dPdW * np.dot(dWdn, dndp)
    return dPdp
