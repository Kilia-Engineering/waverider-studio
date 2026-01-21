from __future__ import annotations
import numpy as np
import pysagas.flow
from pysagas.geometry import Vector
from numpy.typing import ArrayLike
from typing import Union, List, Optional


class Cell:
    """A triangular cell object.

    Attributes
    -----------
    p0 : Vector
        The first vertex of the cell.

    p1 : Vector
        The second vertex of the cell.

    p2 : Vector
        The third vertex of the cell.

    A : float
        The cell face area.

    n : Vector
        The cell normal.

    c : Vector
        The cell centroid.

    dndv : np.array
        The sensitivity of the cell's normal vector to each vertex.

    dAdv : np.array
        The sensitivity of the cell's area to each vertex.

    dvdp : np.array
        The sensitivity of the cell's vertices to each geometric parameter.

    dndp : np.array
        The sensitivity of the cell's normal vector to each geometric parameter.

    dAdp : np.array
        The sensitivity of the cell's area to each geometric parameter.

    dcdp : np.array
        The sensitivity of the centroid to each geometric parameter.

    flowstate : FlowState
        The flowstate associated with the cell.

    sensitivities : np.array
        An array containing the [x,y,z] force sensitivities of the cell.
    """

    def __init__(
        self, p0: Vector, p1: Vector, p2: Vector, face_ids: Optional[list[int]] = None
    ):
        """Constructs a cell, defined by three points.

        Parameters
        ----------
        p0 : Vector
            The first point defining the cell.

        p1 : Vector
            The second point defining the cell.

        p2 : Vector
            The third point defining the cell.
        """
        # Save points
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

        # Save connectivity information (PyMesh)
        self._face_ids = face_ids

        # Calculate cell properites
        self.n = -self.calc_normal(p0, p1, p2)
        self.A = self.calc_area(p0, p1, p2)
        self.c = self.calc_centroid(p0, p1, p2)

        # Calculate geometric sensitivities
        self._dndv: ArrayLike = None
        self._dAdv: ArrayLike = None
        self._dcdv: ArrayLike = None

        # Parameter sensitivities
        self.dvdp: ArrayLike = None  # vertex-parameter sensitivities
        self.dndp: ArrayLike = None  # normal-parameter sensitivities
        self.dAdp: ArrayLike = None  # area-parameter sensitivities
        self.dcdp: ArrayLike = None  # centroid-parameter sensitivities

        # FlowState
        self.flowstate: pysagas.flow.FlowState = None

        # Sensitivities
        self.sensitivities = None
        self.moment_sensitivities = None

        # Cell attributes
        self.attributes = {}

    @property
    def dndv(self):
        if self._dndv is not None:
            # Already exists, return it
            return self._dndv
        else:
            # Calculate and return it
            self._dndv = self.n_sensitivity(self.p0, self.p1, self.p2)
            return self._dndv

    @property
    def dAdv(self):
        if self._dAdv is not None:
            # Already exists, return it
            return self._dAdv
        else:
            # Calculate and return it
            self._dAdv = self.A_sensitivity(self.p0, self.p1, self.p2)
            return self._dAdv

    @property
    def dcdv(self):
        if self._dcdv is not None:
            # Already exists, return it
            return self._dcdv
        else:
            # Calculate and return it
            self._dcdv = self.c_sensitivity(self.p0, self.p1, self.p2)
            return self._dcdv

    @property
    def vertices(self):
        """The cell vertices."""
        return np.array([getattr(self, p).vec for p in ["p0", "p1", "p2"]])

    def to_dict(self):
        """Returns the Cell as a dictionary."""
        # TODO - need to think about representing the cells
        # vs. the points
        # The values at the points will need to be averaged
        # across cells
        pass

    @classmethod
    def from_points(
        cls, points: Union[List[Vector], np.array[Vector]], **kwargs
    ) -> Cell:
        """Constructs a Vector object from an array of coordinates.

        Parameters
        ----------
        points : Union[List[Vector], np.array[Vector]]
            The points defining the cell.

        Returns
        -------
        Cell
        """
        return cls(*points, **kwargs)

    def __repr__(self) -> str:
        return f"Cell({self.p0}, {self.p1}, {self.p2})"

    def __str__(self) -> str:
        return "A Cell"

    def _add_sensitivities(self, dvdp: np.array) -> None:
        """Adds the cell sensitivities to the cell.

        Parameters
        -----------
        dvdp : array
            The sensitivity of each vertex x, y and z, with respect to each
            parameter. The dimensions of dvdp are (9, p), for p parameters.
            Each 3 rows corresponds to the x, y and z sensitivities for each
            vertex of the cell. Each column is for the relevant parameter.
        """
        self.dvdp = dvdp
        self.dndp = np.dot(self.dndv, self.dvdp)
        self.dAdp = np.dot(self.dAdv, self.dvdp)
        self.dcdp = np.dot(self.dcdv, self.dvdp)

    @staticmethod
    def calc_normal(p0: Vector, p1: Vector, p2: Vector) -> Vector:
        """Calculates the normal vector of a cell defined by three points.

        Parameters
        ----------
        p0 : Vector
            The first point defining the cell.

        p1 : Vector
            The second point defining the cell.

        p2 : Vector
            The third point defining the cell.

        Returns
        --------
        normal : Vector
            The unit normal vector of the cell defined by the points.

        References
        -----------
        https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal
        """
        # Define cell edges
        A = p1 - p0
        B = p2 - p0

        # Calculate normal components
        Nx = A.y * B.z - A.z * B.y
        Ny = A.z * B.x - A.x * B.z
        Nz = A.x * B.y - A.y * B.x

        # Construct normal vector
        normal = Vector(x=Nx, y=Ny, z=Nz)

        if normal.norm == 0:
            # Degenerate cell
            raise DegenerateCell()

        # Convert to unit vector
        unit_normal = normal / np.linalg.norm(normal.vec)

        return unit_normal

    @staticmethod
    def calc_area(p0: Vector, p1: Vector, p2: Vector) -> float:
        """Calculates the area of a cell defined by three points.

        Parameters
        ----------
        p0 : Vector
            The first point defining the cell.

        p1 : Vector
            The second point defining the cell.

        p2 : Vector
            The third point defining the cell.

        Returns
        --------
        area : float
            The area of the cell defined by the points.

        References
        -----------
        https://en.wikipedia.org/wiki/Cross_product
        """
        # Define cell edges
        A = p1 - p0
        B = p2 - p0

        # Calculate area
        S = 0.5 * np.linalg.norm(np.cross(A.vec, B.vec))
        return S

    @staticmethod
    def calc_centroid(p0: Vector, p1: Vector, p2: Vector) -> Vector:
        """Calculates the centroid of a cell defined by three points.

        Parameters
        ----------
        p0 : Vector
            The first point defining the cell.

        p1 : Vector
            The second point defining the cell.

        p2 : Vector
            The third point defining the cell.

        Returns
        --------
        c : Vector
            The centroid of the cell defined by the points.

        References
        -----------
        https://en.wikipedia.org/wiki/Centroid
        """
        cx = (p0.x + p1.x + p2.x) / 3
        cy = (p0.y + p1.y + p2.y) / 3
        cz = (p0.z + p1.z + p2.z) / 3
        return Vector(cx, cy, cz)

    @staticmethod
    def n_sensitivity(p0: Vector, p1: Vector, p2: Vector) -> np.array:
        """Calculates the sensitivity of a cell's normal vector
        to the points defining the cell analytically.

        Parameters
        ----------
        p0 : Vector
            The first point defining the cell.

        p1 : Vector
            The second point defining the cell.

        p2 : Vector
            The third point defining the cell.

        Returns
        -------
        sensitivity : np.array
            The sensitivity matrix with size m x n x p. Rows m refer to
            the vertices, columns n refer to the vertex coordinates, and
            slices p refer to the components of the normal vector.
        """
        g = (
            ((p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)) ** 2
            + ((p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)) ** 2
            + ((p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)) ** 2
        )

        # n = np.empty(3)
        # A = p0
        # B = p1
        # C = p2
        # n[0] = ((B.y-A.y)*(C.z-A.z) - (B.z-A.z)*(C.y-A.y)) / g**0.5
        # n[1] = ((B.z-A.z)*(C.x-A.x) - (B.x-A.x)*(C.z-A.z)) / g**0.5
        # n[2] = ((B.x-A.x)*(C.y-A.y) - (B.y-A.y)*(C.x-A.x)) / g**0.5

        M_sense = np.empty((3, 9))
        for row in range(3):
            if row == 0:
                f = (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)  # f_x
            elif row == 1:
                f = (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)  # f_y
            elif row == 2:
                f = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)  # f_z

            for col in range(9):
                if col == 0:  # d/dp0.x
                    g_dash = 2 * (
                        (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                    ) * (-(p1.z - p0.z) + (p2.z - p0.z)) + 2 * (
                        (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                    ) * (
                        -(p2.y - p0.y) + (p1.y - p0.y)
                    )
                elif col == 1:  # d/dp0.y
                    g_dash = 2 * (
                        (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                    ) * (-(p2.z - p0.z) + (p1.z - p0.z)) + 2 * (
                        (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                    ) * (
                        -(p1.x - p0.x) + (p2.x - p0.x)
                    )
                elif col == 2:  # d/dp0.z
                    g_dash = 2 * (
                        (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                    ) * (-(p1.y - p0.y) + (p2.y - p0.y)) + 2 * (
                        (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                    ) * (
                        -(p2.x - p0.x) + (p1.x - p0.x)
                    )
                elif col == 3:  # d/dp1.x
                    g_dash = 2 * (
                        (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                    ) * (-(p2.z - p0.z)) + 2 * (
                        (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                    ) * (
                        (p2.y - p0.y)
                    )
                elif col == 4:  # d/dp1.y
                    g_dash = 2 * (
                        (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                    ) * ((p2.z - p0.z)) + 2 * (
                        (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                    ) * (
                        -(p2.x - p0.x)
                    )
                elif col == 5:  # d/dp1.z
                    g_dash = 2 * (
                        (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                    ) * (-(p2.y - p0.y)) + 2 * (
                        (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                    ) * (
                        (p2.x - p0.x)
                    )
                elif col == 6:  # d/dp2.x
                    g_dash = 2 * (
                        (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                    ) * ((p1.z - p0.z)) + 2 * (
                        (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                    ) * (
                        -(p1.y - p0.y)
                    )
                elif col == 7:  # d/dp2.y
                    g_dash = 2 * (
                        (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                    ) * (-(p1.z - p0.z)) + 2 * (
                        (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                    ) * (
                        (p1.x - p0.x)
                    )
                elif col == 8:  # d/dp2.z
                    g_dash = 2 * (
                        (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                    ) * ((p1.y - p0.y)) + 2 * (
                        (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                    ) * (
                        -(p1.x - p0.x)
                    )

                if row == 0:  # for n..x
                    # ((p1.y-p0.y)*(p2.z-p0.z) - (p1.z-p0.z)*(p2.y-p0.y))  # f_x
                    if col == 1:  # d/dp0.y
                        f_dash = -(p2.z - p0.z) + (p1.z - p0.z)
                    elif col == 2:  # d/dp0.z
                        f_dash = -(p1.y - p0.y) + (p2.y - p0.y)
                    elif col == 4:  # d/dp1.y
                        f_dash = p2.z - p0.z
                    elif col == 5:  # d/dp1.z
                        f_dash = -(p2.y - p0.y)
                    elif col == 7:  # d/dp2.y
                        f_dash = -(p1.z - p0.z)
                    elif col == 8:  # d/dp2.z
                        f_dash = p1.y - p0.y
                    else:
                        f_dash = 0
                if row == 1:  # for n.y  # CHECK
                    # f = ((p1.z-p0.z)*(p2.x-p0.x) - (p1.x-p0.x)*(p2.z-p0.z))  # f_y
                    if col == 0:  # d/dp0.x
                        f_dash = -(p1.z - p0.z) + (p2.z - p0.z)
                    elif col == 2:  # d/dp0.z
                        f_dash = -(p2.x - p0.x) + (p1.x - p0.x)  # CHECK
                    elif col == 3:  # d/dp1.x
                        f_dash = -(p2.z - p0.z)
                    elif col == 5:  # d/dp1.z
                        f_dash = p2.x - p0.x
                    elif col == 6:  # d/dp2.x
                        f_dash = p1.z - p0.z
                    elif col == 8:  # d/dp2.z
                        f_dash = -(p1.x - p0.x)
                    else:
                        f_dash = 0
                if row == 2:  # for n.z  # CHECK
                    # f = ((p1.x-p0.x)*(p2.y-p0.y) - (p1.y-p0.y)*(p2.x-p0.x))  # f_z
                    if col == 0:  # d/dp0.x
                        f_dash = -(p2.y - p0.y) + (p1.y - p0.y)
                    elif col == 1:  # d/dp0.y
                        f_dash = -(p1.x - p0.x) + (p2.x - p0.x)
                    elif col == 3:  # d/dp1.x
                        f_dash = p2.y - p0.y
                    elif col == 4:  # d/dp1.y
                        f_dash = -(p2.x - p0.x)
                    elif col == 6:  # d/dp2.x
                        f_dash = -(p1.y - p0.y)
                    elif col == 7:  # d/dp2.y
                        f_dash = p1.x - p0.x
                    else:
                        f_dash = 0

                # Assign sensitivity
                M_sense[row, col] = (
                    f_dash * (g ** (-0.5)) + f * (-1 / 2) * g ** (-3 / 2) * g_dash
                )
        return M_sense

    @staticmethod
    def A_sensitivity(p0: Vector, p1: Vector, p2: Vector) -> np.array:
        """Calculates the sensitivity of a cell's area to the points
        defining the cell analytically.

        Parameters
        ----------
        p0 : Vector
            The first point defining the cell.

        p1 : Vector
            The second point defining the cell.

        p2 : Vector
            The third point defining the cell.

        Returns
        -------
        sensitivity : np.array
            The sensitivity matrix with size m x n. Rows m refer to
            the vertices, columns n refer to the vertex coordinates.
        """
        g = (
            ((p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)) ** 2
            + ((p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)) ** 2
            + ((p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)) ** 2
        )

        A_sense = np.empty(9)
        for col in range(9):
            if col == 0:  # d/dp0.x
                g_dash = 2 * (
                    (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                ) * (-(p1.z - p0.z) + (p2.z - p0.z)) + 2 * (
                    (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                ) * (
                    -(p2.y - p0.y) + (p1.y - p0.y)
                )
            elif col == 1:  # d/dp0.y
                g_dash = 2 * (
                    (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                ) * (-(p2.z - p0.z) + (p1.z - p0.z)) + 2 * (
                    (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                ) * (
                    -(p1.x - p0.x) + (p2.x - p0.x)
                )
            elif col == 2:  # d/dp0.z
                g_dash = 2 * (
                    (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                ) * (-(p1.y - p0.y) + (p2.y - p0.y)) + 2 * (
                    (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                ) * (
                    -(p2.x - p0.x) + (p1.x - p0.x)
                )
            elif col == 3:  # d/dp1.x
                g_dash = 2 * (
                    (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                ) * (-(p2.z - p0.z)) + 2 * (
                    (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                ) * (
                    (p2.y - p0.y)
                )
            elif col == 4:  # d/dp1.y
                g_dash = 2 * (
                    (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                ) * ((p2.z - p0.z)) + 2 * (
                    (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                ) * (
                    -(p2.x - p0.x)
                )
            elif col == 5:  # d/dp1.z
                g_dash = 2 * (
                    (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                ) * (-(p2.y - p0.y)) + 2 * (
                    (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                ) * (
                    (p2.x - p0.x)
                )
            elif col == 6:  # d/dp2.x
                g_dash = 2 * (
                    (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                ) * ((p1.z - p0.z)) + 2 * (
                    (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                ) * (
                    -(p1.y - p0.y)
                )
            elif col == 7:  # d/dp2.y
                g_dash = 2 * (
                    (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                ) * (-(p1.z - p0.z)) + 2 * (
                    (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)
                ) * (
                    (p1.x - p0.x)
                )
            elif col == 8:  # d/dp2.z
                g_dash = 2 * (
                    (p1.y - p0.y) * (p2.z - p0.z) - (p1.z - p0.z) * (p2.y - p0.y)
                ) * ((p1.y - p0.y)) + 2 * (
                    (p1.z - p0.z) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.z - p0.z)
                ) * (
                    -(p1.x - p0.x)
                )

            A_sense[col] = 0.5 * (1 / 2) * g ** (-1 / 2) * g_dash

        return A_sense

    @staticmethod
    def c_sensitivity(p0: Vector, p1: Vector, p2: Vector) -> np.array:
        """Calculates the sensitivity of a cell's centroid to the
        points defining the cell.

        Parameters
        ----------
        p0 : Vector
            The first point defining the cell.

        p1 : Vector
            The second point defining the cell.

        p2 : Vector
            The third point defining the cell.

        Returns
        -------
        sensitivity : np.array
            The sensitivity matrix with size m x n x p. Rows m refer to
            the vertices, columns n refer to the vertex coordinates, and
            slices p refer to the components of the centroid point.
        """
        c_sens = np.array(
            [
                [1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3, 0, 0],
                [0, 1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3, 0],
                [0, 0, 1 / 3, 0, 0, 1 / 3, 0, 0, 1 / 3],
            ]
        )
        return c_sens


class DegenerateCell(Exception):
    """Exception raised for degenerate cells."""

    def __init__(self, message="Degenerate cell"):
        self.message = message
        super().__init__(self.message)
