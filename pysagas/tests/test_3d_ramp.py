import numpy as np

np.seterr(all="ignore")

from pysagas.geometry import Vector, Cell
from pysagas.flow import GasState, FlowState
from pysagas.cfd.oblique_prandtl_meyer import OPM
from pysagas.sensitivity.calculator import SensitivityCalculator


def calculate_pressures(flow: FlowState, theta: float) -> float:
    """Calculates the pressure from a flow state and deflecion angle
    using ideal gas oblique shock theory.

    Parameters
    ----------
    flow : FlowState
        The flow state.

    theta : float
        The deflection angle (radians).

    Returns
    --------
    P2 : float
        The pressure behind the oblique shock.
    """
    beta = OPM.oblique_beta(
        M1=flow.M, theta=abs(theta), gamma=flow.gamma, tolerance=1.0e-6
    )
    P2_P1 = OPM.oblique_p2_p1(flow.M, beta, gamma=flow.gamma)
    P2 = P2_P1 * flow.P
    return P2


def run_main():
    # Define freestream flow state
    freestream = GasState(mach=6, pressure=700, temperature=70)

    # Define geometric parameters
    theta = np.radians(10)
    length = 0.01
    width = 0.02
    parameters = [theta, length, width]

    beta = OPM.oblique_beta(
        freestream.M, abs(theta), gamma=freestream.gamma, tolerance=1.0e-6
    )
    M_ramp = OPM.oblique_M2(freestream.M, beta, theta, gamma=freestream.gamma)
    T2_T1 = OPM.oblique_T2_T1(freestream.M, beta, gamma=freestream.gamma)

    P_ramp = calculate_pressures(flow=freestream, theta=theta)

    T_ramp = T2_T1 * freestream.T
    rho_ramp = P_ramp / (freestream.R * T_ramp)
    a_ramp = np.sqrt(freestream.gamma * freestream.R * T_ramp)
    vel_ramp = M_ramp * a_ramp

    # Step 1: Work out geometry position sensitivities
    #     p00-------p10
    #      |  A   /  |
    #      |    /    |        yo-->x
    #      |  /   B  |         |
    #     p01-------p11       \/z

    p00 = Vector(0.0, 0.0, 0.0)
    p01 = Vector(0.0, 0.0, parameters[2])
    p10 = Vector(
        parameters[1] * np.cos(parameters[0]),
        parameters[1] * np.sin(parameters[0]),
        0.0,
    )
    p11 = Vector(
        parameters[1] * np.cos(parameters[0]),
        parameters[1] * np.sin(parameters[0]),
        parameters[2],
    )

    # Calculate cell point-to-parameter sensitivities
    p00_sense = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    p01_sense = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p10_sense = np.array(
        [
            [parameters[1] * -np.sin(parameters[0]), np.cos(parameters[0]), 0.0],
            [parameters[1] * np.cos(parameters[0]), np.sin(parameters[0]), 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    p11_sense = np.array(
        [
            [parameters[1] * -np.sin(parameters[0]), np.cos(parameters[0]), 0.0],
            [parameters[1] * np.cos(parameters[0]), np.sin(parameters[0]), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Cell vertex sensitivity to params (dv/dp)
    pos_sense_A = np.vstack((p00_sense, p01_sense, p10_sense))
    pos_sense_B = np.vstack((p01_sense, p11_sense, p10_sense))

    # Define cells
    A = Cell(p00, p01, p10)
    B = Cell(p01, p11, p10)

    # Add sensitivities
    A._add_sensitivities(pos_sense_A)
    B._add_sensitivities(pos_sense_B)

    # Calculate velocity vector
    vel_ramp_vector = [vel_ramp * np.cos(theta), vel_ramp * np.sin(theta), 0.0]

    # Define ramp flow state
    ramp_velocity_vector = Vector.from_coordinates(vel_ramp_vector)
    ramp_fs = FlowState(
        mach=M_ramp, pressure=P_ramp, temperature=T_ramp, direction=ramp_velocity_vector
    )

    # Attribute the ramp flow state to each Cell
    A.flowstate = ramp_fs
    B.flowstate = ramp_fs

    # Calculate force sensitivity
    F_sense, _ = SensitivityCalculator.net_sensitivity(cells=[A, B])

    # Calculate error with finite differencing method
    fd_F_sense = calc_fd_sens(parameters, freestream)
    errors = 100 * np.nan_to_num(
        np.divide((fd_F_sense.T - F_sense), fd_F_sense.T), posinf=0, neginf=0
    )

    # Test against expected values (error < 10%)
    assert np.max(np.abs(errors)) < 10, "Adjoints inaccurately calculated"


def calculate_force_vector(P: float, n: np.array, A: float) -> np.array:
    """Calculates the force vector components, acting on a
    surface defined by its area and normal vector.

    Parameters
    ----------
    P : float
        The pressure (Pa).

    n : np.array
        The normal vector.

    A : float
        The reference area (m^2).

    Returns
    --------
    forces : np.array
        The force components.
    """
    F_x = A * P * np.dot(n, np.array([-1, 0, 0]))
    F_y = A * P * np.dot(n, np.array([0, -1, 0]))
    F_z = A * P * np.dot(n, np.array([0, 0, -1]))

    return [F_x, F_y, F_z]


def calc_fd_sens(parameters: list, freestream: GasState):
    # Perturbation parameters
    dp = 0.001  # 0.1% perturbation

    # Calculate sensitivities using finite difference
    pressure_sensitivities_finitediff = np.zeros((1, 3))
    force_sensitivities_finitediff = np.zeros((3, 3))
    for i, _ in enumerate(parameters):
        p_high = parameters.copy()
        p_low = parameters.copy()
        delta_p = dp * abs(parameters[i])
        p_high[i] = p_high[i] + delta_p
        p_low[i] = p_low[i] - delta_p

        pressure_high = calculate_pressures(freestream, p_high[0])
        pressure_low = calculate_pressures(freestream, p_low[0])

        # Pressure sensitivities
        pressure_sensitivities_finitediff[:, i] = (pressure_high - pressure_low) / (
            2 * delta_p
        )

        # Force sensitivities
        n_ramp_high = np.array([-1 * np.sin(p_high[0]), np.cos(p_high[0]), 0])
        n_ramp_low = np.array([-1 * np.sin(p_low[0]), np.cos(p_low[0]), 0])
        area_high = p_high[1] * p_high[2]
        area_low = p_low[1] * p_low[2]

        forces_high = calculate_force_vector(
            P=pressure_high, n=n_ramp_high, A=area_high
        )
        forces_low = calculate_force_vector(P=pressure_low, n=n_ramp_low, A=area_low)

        force_sensitivities_finitediff[:, i] = [
            (fh - fl) / (2 * delta_p) for fh, fl in zip(forces_high, forces_low)
        ]

    return force_sensitivities_finitediff


def test_ramp():
    run_main()


if __name__ == "__main__":
    run_main()
