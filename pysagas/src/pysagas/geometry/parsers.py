import pandas as pd
from stl import mesh
from tqdm import tqdm
import multiprocess as mp
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pysagas.geometry import Cell, Vector
from pysagas.utilities import add_sens_data


class AbstractParser(ABC):
    """Interface for a geometry parser."""

    filetype = None

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"PySAGAS {self.filetype} parser"

    def __str__(self) -> str:
        return f"PySAGAS {self.filetype} parser"

    @property
    @abstractmethod
    def filetype(self):
        # This is a placeholder for a class variable defining the parser file type
        pass

    @abstractmethod
    def load(self) -> List[Cell]:
        """Load cells from file."""

    @classmethod
    @abstractmethod
    def load_from_file(self) -> List[Cell]:
        """Convenience method for loading cells from file."""


class Parser(AbstractParser):
    def __init__(self, filepath: str, verbosity: int = 1) -> None:
        self.filepath = filepath
        self.verbosity = verbosity

    @classmethod
    def load_from_file(
        cls,
        filepath: str,
        geom_sensitivities: Optional[Union[str, pd.DataFrame]] = None,
        verbosity: Optional[int] = 1,
        **kwargs,
    ) -> List[Cell]:
        """Convenience method for loading cells from file.

        Parameters
        ----------
        filepath : str
            The filepath to the geometry.

        geom_sensitivities : str | DataFrame, optional
            The geometry sensitivity data, to optionally add to the loaded cells. This can
            be provided as a path to the data in csv format, or directly as a Pandas
            DataFrame. The default is None.

        verbosity : int, optional
            The verbosity of the code. The defualt is 1.

        **kwargs
            Additional keyword arguments can be provided to control the sensitivity matching
            algorithm.

        See Also
        --------
        pysagas.utilities.add_sens_data
        """

        # Create parser instance
        parser = cls(filepath, verbosity)

        # Load file
        cells = parser.load()

        if geom_sensitivities:
            # Check input type
            if isinstance(geom_sensitivities, str):
                # File path provided, load into dataframe
                geom_sensitivities = pd.read_csv(geom_sensitivities)

            elif not isinstance(geom_sensitivities, pd.DataFrame):
                raise TypeError("Invalid data provided for 'geom_sensitivities'.")

            # Add sensitivity data to cells
            add_sens_data(
                cells=cells,
                data=geom_sensitivities,
                verbosity=verbosity,
                **kwargs,
            )
        return cells


class STL(Parser):
    filetype = "STL"

    def load(self) -> List[Cell]:
        # Load the STL
        mesh_obj = mesh.Mesh.from_file(self.filepath)

        cells = []
        # TODO - can face ids be inferred?
        if self.verbosity > 0:
            print("\nTranscribing cells:")
            pbar = tqdm(
                total=len(mesh_obj.vectors),
                position=0,
                leave=True,
                desc="  Cell transcription progress",
            )

        for vector_triple in mesh_obj.vectors:
            vertices = [Vector.from_coordinates(v) for v in vector_triple]
            try:
                cell = Cell.from_points(vertices)
                cells.append(cell)
            except:
                pass

            # Update progress bar
            if self.verbosity > 0:
                pbar.update(1)

        if self.verbosity > 0:
            pbar.close()
            print("Done.")

        return cells


class MeshIO(Parser):
    """Meshio parser"""

    filetype = "meshio mesh"

    def __init__(self, filepath: str, verbosity: int = 1) -> None:
        # Import meshio
        try:
            import meshio
        except ModuleNotFoundError:
            raise Exception("Could not find meshio. Please install it and try again.")
        self._meshio = meshio
        super().__init__(filepath, verbosity)

    def load(self) -> List[Cell]:
        def mp_wrapper(face):
            vertices = [Vector.from_coordinates(mesh_vertices[i]) for i in face]
            try:
                cell = Cell.from_points(vertices, face_ids=face)
            except:
                cell = None
            return cell

        # Load the STL
        mesh_obj = self._meshio.read(self.filepath)

        if self.verbosity > 0:
            print("\nTranscribing cells.")

        # Create multiprocessing pool to construct cells
        cells = []
        pool = mp.Pool()
        mesh_vertices = mesh_obj.points
        for result in pool.map(mp_wrapper, mesh_obj.cells[0].data):
            if result is not None:
                cells.append(result)

        if self.verbosity > 0:
            print("Done.")

        return cells


class PyMesh(Parser):
    filetype = "PyMesh STL"

    def __init__(self, filepath: str, verbosity: int = 1) -> None:
        # Import PyMesh
        try:
            import pymesh
        except ModuleNotFoundError:
            raise Exception(
                "Could not find pymesh. Please follow the "
                + "installation instructions at "
                + "https://pymesh.readthedocs.io/en/latest/installation.html"
            )
        self._pymesh = pymesh
        super().__init__(filepath, verbosity)

    def load(self) -> List[Cell]:
        def mp_wrapper(face):
            vertices = [Vector.from_coordinates(mesh_vertices[i]) for i in face]
            try:
                cell = Cell.from_points(vertices, face_ids=face)
            except:
                cell = None
            return cell

        # Load the STL
        mesh_obj = self._pymesh.load_mesh(self.filepath)

        if self.verbosity > 0:
            print("\nTranscribing cells.")

        # Create multiprocessing pool to construct cells
        cells = []
        pool = mp.Pool()
        mesh_vertices = mesh_obj.vertices
        for result in pool.map(mp_wrapper, mesh_obj.faces):
            if result is not None:
                cells.append(result)

        if self.verbosity > 0:
            print("Done.")

        return cells


class TRI(Parser):
    filetype = ".tri"

    def load(self) -> List[Cell]:
        # Parse .tri file
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        grid = root[0]
        piece = grid[0]
        points = piece[0]
        cells = piece[1]

        points_data = points[0].text
        cells_data = cells[0].text

        points_data_list = [el.split() for el in points_data.splitlines()[1:]]
        points_data_list = [[float(j) for j in i] for i in points_data_list]

        cells_data_list = [el.split() for el in cells_data.splitlines()[1:]]
        cells_data_list = [[int(j) for j in i] for i in cells_data_list]

        cells = []
        if self.verbosity > 0:
            print("\nTranscribing cells:")
            pbar = tqdm(
                total=len(cells_data_list),
                position=0,
                leave=True,
                desc="  Cell transcription progress",
            )
        for vertex_idxs in cells_data_list:
            vertices = [
                Vector.from_coordinates(points_data_list[i]) for i in vertex_idxs
            ]
            cell = Cell.from_points(vertices, face_ids=vertex_idxs)
            cells.append(cell)

            # Update progress bar
            if self.verbosity > 0:
                pbar.update(1)

        if self.verbosity > 0:
            pbar.close()
            print("Done.")

        return cells

    @classmethod
    def load_from_file(cls, filepath: str, verbosity: int = 1, **kwargs) -> List[Cell]:
        """Convenience method for loading cells from file."""
        # Create parser instance
        parser = cls(filepath, verbosity)

        # Load file
        cells = parser.load()

        return cells
