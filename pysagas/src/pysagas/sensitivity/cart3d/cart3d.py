import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
from pysagas.flow import FlowState
from pysagas.sensitivity.calculator import SensitivityCalculator
from pysagas.geometry import Vector, Cell, DegenerateCell
from pysagas.sensitivity.cart3d.utilities import process_components_file


class Cart3DSensitivityCalculator(SensitivityCalculator):
    """PySAGAS Cart3D flow sensitivity calculator."""

    solver = "Cart3D"

    def __init__(
        self,
        freestream: FlowState,
        sensitivity_filepath: str,
        components_filepath: Optional[str] = None,
        pointdata: Optional[pd.DataFrame] = None,
        celldata: Optional[pd.DataFrame] = None,
        write_data: Optional[bool] = False,
        verbosity: Optional[int] = 1,
        **kwargs,
    ) -> None:
        """A PySAGAS sensitivity calculator for Cart3D.

        Parameters
        ----------
        freestream : FlowState
            The flow state of the freestream.

        sensitivity_filepath : str
            The filepath to the geometry sensitivities.

        components_filepath : str, optional
            The filepath to the Components.i.plt file to be processed.
            The default is None.

        pointdata : pd.DataFrame, optional
            The point data. Must be supplied with celldata.

        celldata : pd.DataFrame, optional
            The cell data. Must be supplied with pointdata.

        write_data : bool, optional
            Write the flow data to CSV files. The default is True.

        verbosity : int, optional
            The verbosity of the code. The defualt is 1.
        """
        # Load data
        self.sensdata = pd.read_csv(sensitivity_filepath)
        if pointdata is not None and celldata is not None:
            self.pointdata = pointdata
            self.celldata = celldata
        else:
            self.pointdata, self.celldata = process_components_file(
                freestream.a,
                freestream.rho,
                components_filepath,
                write_data=write_data,
                verbosity=verbosity,
            )
        self.verbosity = verbosity

        # Check dimensionality
        l1 = len(self.sensdata)
        l2 = len(self.pointdata)
        if l1 != l2:
            raise ValueError(
                f"The sensitivity data does not match the point data ({l1} vs. {l2})."
            )

        super().__init__(**kwargs)

    def _transcribe_cells(self, parameters: List[str]) -> List[Cell]:
        """Transcribes the cells from Components.i.plt files into
        PySAGAS Cell objects.

        Parameters
        -----------
        parameters : List[str]
            A list of the geometric design parameters.

        Returns
        --------
        cells : List[Cell]
            A list of all transcribed cells.

        See Also
        --------
        pysagas.geometry.Cell
        """
        coordinates = self.pointdata[["Points_0", "Points_1", "Points_2"]]
        cell_vertex_ids = self.celldata[
            ["Point Index 0", "Point Index 1", "Point Index 2"]
        ]

        # Construct cells
        if self.verbosity > 0:
            print("\nTranscribing cells from Components.i.plt...")
            pbar = tqdm(
                total=len(self.celldata.index),
                position=0,
                leave=True,
            )
        cells = []
        degen_cells = 0
        for cell in self.celldata.index:
            vertex_ids = cell_vertex_ids.loc[cell].values
            cell_v_coords = [coordinates.loc[v_id].values for v_id in vertex_ids]
            vertices = [Vector.from_coordinates(c) for c in cell_v_coords]

            # Extract sensitivity information for the cell
            vertex_sensitivity_info = [self.sensdata.loc[v_id] for v_id in vertex_ids]

            dvdp = np.empty((9, len(parameters)))
            for i, p in enumerate(parameters):
                r = 0
                for v in vertex_sensitivity_info:
                    for c in ["x", "y", "z"]:
                        dvdp[r, i] = v[f"d{c}d{p}"]
                        r += 1

            # Create Cell
            try:
                newcell = Cell.from_points(vertices)
            except DegenerateCell:
                # Bad cell, skip it
                degen_cells += 1
                if self.verbosity > 0:
                    pbar.update(1)

                    if self.verbosity > 2:
                        print("\033[1mWarning\033[0m: Degenerate cell.")
                continue

            # Add geometry sensitivity information
            newcell._add_sensitivities(np.array(dvdp))

            # Add flow state at cell
            cell_velocity = Vector.from_coordinates(
                self.celldata.loc[cell][["U", "V", "W"]].values
            )
            cell_flowstate = FlowState(
                mach=self.celldata.loc[cell]["M"],
                pressure=self.celldata.loc[cell]["p"],
                temperature=self.celldata.loc[cell]["T"],
                direction=cell_velocity,
            )
            newcell.flowstate = cell_flowstate

            # Append new Cell
            cells.append(newcell)

            # Update progress bar
            if self.verbosity > 0:
                pbar.update(1)

        if self.verbosity > 0:
            pbar.close()
            print("Done.")

            if self.verbosity > 1:
                if degen_cells > 0:
                    print(f"{100*degen_cells/len(self.celldata):.2f}% degenerate cells")

        return cells
