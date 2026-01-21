import numpy as np
from pysagas.geometry import Vector


class GasState:
    """An ideal gas state defined by Mach number, pressure and
    temperature.
    """

    gamma = 1.4
    R = 287  # J/kg·K.3

    def __init__(
        self, mach: float, pressure: float, temperature: float, gamma: float = 1.4
    ) -> None:
        """Define a new gas state.

        Parameters
        -----------
        mach : float
            The flow Mach number.

        pressure : float
            The flow pressure (Pa).

        temperature : float
            The flow temperature (K).

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.
        """
        # Assign properties
        self._T = temperature
        self._P = pressure
        self._M = mach
        self._gamma = gamma

    def __str__(self) -> str:
        return f"Mach {self.M} flow condition with P = {self.P}, T = {self.T}."

    def __repr__(self) -> str:
        return f"Flow(M={self.M}, P={self.P}, T={self.T})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GasState):
            raise Exception(f"Cannot compare {type(other)} to GasState.")
        return (
            (self._T == other._T)
            & (self._P == other._P)
            & (self._M == other._M)
            & (self._gamma == other._gamma)
        )

    @property
    def T(self):
        return self._T

    @property
    def P(self):
        return self._P

    @property
    def M(self):
        return self._M

    @property
    def a(self):
        return (self.gamma * self.R * self.T) ** 0.5

    @property
    def rho(self):
        return self.P / (self.R * self.T)

    @property
    def v(self):
        return self.M * self.a

    @property
    def q(self):
        return 0.5 * self.rho * self.v**2

    @property
    def gamma(self):
        return self._gamma


class FlowState(GasState):
    """An ideal gas state defined by Mach number, pressure and
    temperature, with a flow direction.
    """

    def __init__(
        self,
        mach: float,
        pressure: float,
        temperature: float,
        direction: Vector = None,
        aoa: float = 0.0,
        gamma: float = 1.4,
    ) -> None:
        """Define a new flow state.

        Parameters
        -----------
        mach : float
            The flow Mach number.

        pressure : float
            The flow pressure (Pa).

        temperature : float
            The flow temperature (K).

        direction : Vector, optional
            The direction vector of the flow. The default is Vector(1,0,0).

        aoa : float, optional
            The angle of attack of the flow. The default is 0.0 (specified in
            degrees).

        gamma : float, optional
            The ratio of specific heats. The default is 1.4.
        """
        super().__init__(mach, pressure, temperature, gamma)
        if direction:
            # Use direction provided
            self.direction = direction.unit
        else:
            # Use AoA to calculate direction
            self.direction = Vector(1, 1 * np.tan(np.deg2rad(aoa)), 0).unit

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlowState):
            raise Exception(f"Cannot compare {type(other)} to FlowState.")
        same_gs = super().__eq__(other)
        return same_gs & (self.direction == other.direction)

    @property
    def vx(self):
        return self.Vector.x

    @property
    def vy(self):
        return self.Vector.y

    @property
    def vz(self):
        return self.Vector.z

    @property
    def vec(self):
        return self.Vector.vec

    @property
    def Vector(self) -> Vector:
        return self.direction * self.v

    @property
    def aoa(self):
        aoa = np.rad2deg(np.arctan(self.vec[1] / self.vec[0]))
        return round(aoa, 6)


if __name__ == "__main__":
    flow = FlowState(6, 700, 70)
