import numpy as np
import pandas as pd
from pysagas.flow import FlowState
from abc import ABC, abstractmethod
from pysagas.geometry import Vector, Cell
from pysagas.utilities import add_sens_data
from pysagas.cfd.solver import SensitivityResults
from typing import List, Callable, Tuple, Optional, Union, Literal
from pysagas.sensitivity.models import (
    piston_sensitivity,
    van_dyke_sensitivity,
    isentropic_sensitivity,
)


class AbstractSensitivityCalculator(ABC):
    """Abstract sensitivity calculator class defining the interface."""

    solver = None

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"PySAGAS Sensitivity Calculator for {self.solver}"

    def __str__(self) -> str:
        return f"PySAGAS Sensitivity Calculator for {self.solver}"

    @property
    @abstractmethod
    def solver(self):
        # This is a placeholder for a class variable defining the Sensitivity Calculator's solver
        pass

    @abstractmethod
    def _transcribe_cells(self, parameters: List):
        """Transcribes the unified Cells from the solver data.

        Returns
        --------
        cells : List[Cell]
            A list of the Cells from the flow solution.
        """
        # This method should be overloaded with the solver-specific method
        pass

    @abstractmethod
    def _extract_parameters(self) -> List[str]:
        """Extract the geometry parameters.

        Returns
        -------
        parameters : List[str]
            A list of the parameter names.
        """
        # This method should be overloaded with the solver-specific method
        pass

    @abstractmethod
    def to_csv(self):
        """Dumps the sensitivity data to CSV file."""
        pass


class SensitivityCalculator(AbstractSensitivityCalculator):
    """SensitivityCalculator base class."""

    def __init__(self, **kwargs) -> None:
        self.cells: list[Cell] = None

    def calculate(
        self,
        sensitivity_model: Optional[
            Union[Callable, Literal["piston", "van_dyke", "isentropic"]]
        ] = "van_dyke",
        cog: Optional[Vector] = Vector(0, 0, 0),
        flowstate: Optional[FlowState] = None,
        **kwargs,
    ) -> SensitivityResults:
        """Calculate the force sensitivities of the surface to the
        parameters.

        Parameters
        ----------
        sensitivity_model : Callable | Literal["piston", "van_dyke", "isentropic"], optional
            The model used to calculate the pressure/parameter sensitivities.
            The default is Van Dyke's second-order theory model.

        cog : Vector, optional
            The centre of gravity. The default is Vector(0, 0, 0).

        flowstate : FlowState, optional
            The flowstate associated with this result. The default is None.

        Returns
        -------
        SensitivityResults : the sensitivity results object.
        """
        if self.verbosity > 0:
            print("\nCalculating aerodynamic sensitivities.")

        parameters = self._extract_parameters()

        if self.cells is None:
            self.cells = self._transcribe_cells(parameters=parameters)

        params_sens_cols = []
        for p in parameters:
            for d in ["x", "y", "z"]:
                params_sens_cols.append(f"d{d}d_{p}")

        # Calculate force sensitivity
        F_sense, M_sense = self.net_sensitivity(
            cells=self.cells, sensitivity_model=sensitivity_model, cog=cog, **kwargs
        )

        # Construct dataframes to return
        df_f = pd.DataFrame(
            F_sense, columns=["dFx/dp", "dFy/dp", "dFz/dp"], index=parameters
        )
        df_m = pd.DataFrame(
            M_sense, columns=["dMx/dp", "dMy/dp", "dMz/dp"], index=parameters
        )

        if self.verbosity > 0:
            print("Done.")

        # Construct results
        result = SensitivityResults(
            f_sens=df_f,
            m_sens=df_m,
            freestream=flowstate,
        )

        return result

    def to_csv(self):
        """Dumps the sensitivity data to CSV file."""
        if self.cells is not None:
            pass

        else:
            raise Exception("No cells have been transcribed.")

    def _extract_parameters(self):
        parameters = []
        for e in self.sensdata.columns:
            e: str
            if e.startswith("dxd") or e.startswith("dyd") or e.startswith("dzd"):
                # Sensitivity coluns
                if e[3:] not in parameters:
                    parameters.append(e[3:])
        return parameters

    @staticmethod
    def net_sensitivity(
        cells: List[Cell],
        sensitivity_model: Optional[
            Union[Callable, Literal["piston", "van_dyke", "isentropic"]]
        ] = "van_dyke",
        cog: Vector = Vector(0, 0, 0),
        **kwargs,
    ) -> Tuple[np.array, np.array]:
        """Calcualtes the net force and moment sensitivities for a list of Cells.

        Parameters
        ----------
        cells : list[Cell]
            The cells to be analysed.

        sensitivity_model : Callable | Literal["piston", "van_dyke", "isentropic"], optional
            The model used to calculate the pressure/parameter sensitivities.
            The default is Van Dyke's second-order theory model.

        cog : Vector, optional
            The reference centre of gravity, used in calculating the moment
            sensitivities. The defualt is Vector(0,0,0).

        Returns
        --------
        dFdp : np.array
            The force sensitivity matrix with respect to the parameters.

        dMdp : np.array
            The moment sensitivity matrix with respect to the parameters.
        """
        # Get sensitivity handle
        if isinstance(sensitivity_model, Callable):
            # Use function provided
            sensitivity_function = sensitivity_model
        elif isinstance(sensitivity_model, str):
            # Get function from method specified
            sensitivity_function = globals().get(f"{sensitivity_model}_sensitivity")
            if sensitivity_function is None:
                raise Exception("Invalid sensitivity method specified.")

        dFdp = 0
        dMdp = 0
        for cell in cells:
            # Calculate force sensitivity
            dFdp_c, dMdp_c = SensitivityCalculator.cell_sensitivity(
                cell=cell, sensitivity_function=sensitivity_function, cog=cog, **kwargs
            )

            dFdp += dFdp_c
            dMdp += dMdp_c

        return dFdp, dMdp

    @staticmethod
    def cell_sensitivity(
        cell: Cell,
        sensitivity_function: Callable,
        cog: Vector = Vector(0, 0, 0),
        **kwargs,
    ) -> Tuple[np.array, np.array]:
        """Calculates force and moment sensitivities for a single cell.

        Parameters
        ----------
        cell : Cell
            The cell.

        sensitivity_function : Callable
            The function to use when calculating the pressure sensitivities.

        cog : Vector, optional
            The reference centre of gravity, used in calculating the moment
            sensitivities. The defualt is Vector(0,0,0).

        Returns
        --------
        sensitivities : np.array
            An array of shape n x 3, for a 3-dimensional cell with
            n parameters.

        See Also
        --------
        all_dfdp : a wrapper to calculate force sensitivities for many cells
        """
        # Initialisation
        all_directions = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        sensitivities = np.zeros(shape=(cell.dndp.shape[1], 3))
        moment_sensitivities = np.zeros(shape=(cell.dndp.shape[1], 3))

        # Calculate moment dependencies
        r = cell.c - cog
        F = cell.flowstate.P * cell.A * cell.n.vec

        # For each parameter
        for p_i in range(cell.dndp.shape[1]):
            # Calculate pressure sensitivity
            dPdp = sensitivity_function(cell=cell, p_i=p_i, **kwargs)

            # Evaluate for sensitivities for each direction
            for i, direction in enumerate(all_directions):
                dF = (
                    dPdp * cell.A * np.dot(cell.n.vec, direction.vec)
                    + cell.flowstate.P
                    * cell.dAdp[p_i]
                    * np.dot(cell.n.vec, direction.vec)
                    + cell.flowstate.P
                    * cell.A
                    * np.dot(-cell.dndp[:, p_i], direction.vec)
                )
                sensitivities[p_i, i] = dF

            # Now evaluate moment sensitivities
            moment_sensitivities[p_i, :] = np.cross(
                r.vec, sensitivities[p_i, :]
            ) + np.cross(cell.dcdp[:, p_i], F)

        # Append to cell
        cell.sensitivities = sensitivities
        cell.moment_sensitivities = moment_sensitivities

        return sensitivities, moment_sensitivities


class GenericSensitivityCalculator(SensitivityCalculator):
    solver = "Generic Flow Solver"

    def __init__(
        self,
        cells: List[Cell],
        sensitivity_filepath: Optional[str] = None,
        cells_have_sens_data: Optional[bool] = False,
        verbosity: Optional[int] = 1,
        **kwargs,
    ) -> None:
        """Instantiate a generic PySAGAS sensitivity calculator.

        Parameters
        ----------
        cells : list[Cell]
            A list of transcribed Cell objects, containing the nominal flow solution.

        sensitivity_filepath : str
            The filepath to the geometry sensitivities. This must be provided, if the
            cells do not have geometric sensitivity information attached. This must
            also be provided anyway, to determine the geometric design parameters.

        cells_have_sens_data : bool, optional
            The cells already have geometric sensitivity data matched to them. When this
            is True, the sensitivity_filepath argument (if provided) is ignored. The
            default is False.

        verbosity : int, optional
            The verbosity of the code. The defualt is 1.
        """
        super().__init__(**kwargs)

        if cells_have_sens_data:
            self.cells = cells
        else:
            self._pre_transcribed_cells = cells

        self.sensdata = pd.read_csv(sensitivity_filepath)
        self.verbosity = verbosity

    def _transcribe_cells(self, **kwargs) -> List[Cell]:
        """This is a dummy method to satisfy the abstract base class. Transcribed cells
        are provided upon instantiation of the sensitivity calculator.
        """
        # Add sensitivity data to _pre_transcribed_cells
        add_sens_data(
            cells=self._pre_transcribed_cells,
            data=self.sensdata,
            verbosity=self.verbosity,
        )

        return self._pre_transcribed_cells
