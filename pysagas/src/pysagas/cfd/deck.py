import copy
import pandas as pd
from typing import Optional, Union
from abc import ABC, abstractmethod
from pysagas.cfd.solver import FlowResults, SensitivityResults


class AbstractDeck(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> None:
        pass

    @abstractmethod
    def insert(self):
        """Insert data into the deck."""
        pass

    @abstractmethod
    def to_csv(self):
        """Write the deck to csv."""
        pass

    @abstractmethod
    def from_csv(self, **kwargs):
        """Load the deck from csv."""
        pass

    # @abstractmethod
    # def interpolate(self, **kwargs):
    #     # TODO - implement with scipy.interpolate.interpn
    #     pass


class Deck(AbstractDeck):
    TYPE = "deck"

    def __init__(
        self,
        inputs: list[str],
        columns: list[str],
        **kwargs,
    ) -> None:
        """Instantiate a new deck.

        Parameters
        ----------
        inputs : list[str]
            A list of the inputs to this aerodeck. For example, ["aoa", "mach"].

        columns : list[str]
            A list of column names to use for this deck. For example, ["CL", "CD", "Cm"].
        """
        self._deck = pd.DataFrame(columns=inputs + columns)
        self._inputs = inputs

    def __repr__(self) -> None:
        return f"{self.TYPE}"

    def __str__(self) -> None:
        return self.__repr__()

    @property
    def deck(self) -> pd.DataFrame:
        return self._deck.drop_duplicates()

    def to_csv(self, file_prefix: Optional[str] = None):
        """Save the deck to CSV file.

        Parameters
        ----------
        file_prefix : str, optional
            The CSV file name prefix. If None is provided, the deck __repr__
            will be used. The default is None.
        """
        file_prefix = file_prefix if file_prefix else self.__repr__()
        self.deck.to_csv(f"{file_prefix}.csv", index=False)


class MultiDeck(AbstractDeck):
    """Collection of multiple Deck objects."""

    def __init__(
        self,
        inputs: list[str],
        parameters: list[str],
        base_deck: Deck,
        **kwargs,
    ) -> None:
        """Instantiate a new deck collection.

        Parameters
        ----------
        inputs : list[str]
            A list of the inputs to this deck. For example, ["aoa", "mach"].

        parameters : list[str]
            A list of the parameters to this deck. For example, ["wingspan", "length"].

        base_deck : Deck
            The base deck to initalise self._decks with.
        """
        self._inputs = inputs
        self._parameters = list(parameters)
        self._decks: dict[str, Deck] = {p: copy.deepcopy(base_deck) for p in parameters}


class AeroDeck(Deck):
    """Aerodynamic coefficient deck."""

    TYPE = "aerodeck"

    def __init__(
        self,
        inputs: list[str],
        columns: list[str] = ["CL", "CD", "Cm"],
        a_ref: float = 1,
        c_ref: float = 1,
    ) -> None:
        """Instantiate a new aerodeck.

        Parameters
        ----------
        inputs : list[str]
            A list of the inputs to this aerodeck. For example, ["aoa", "mach"].

        columns : list[str], optional
            A list of column names to use for this deck. The default is ["CL", "CD", "Cm"].

        a_ref : float, optional
            The reference area. The default is 1.

        c_ref : float, optional
            The reference length. The default is 1.
        """

        # Save reference properties
        self._a_ref = a_ref
        self._c_ref = c_ref
        self._columns = columns

        # Complete instantiation
        super().__init__(inputs, columns)

    def insert(self, result: Union[FlowResults, dict[str, float]], **kwargs):
        """Insert aerodynamic coefficients into the aerodeck.

        Parameters
        ----------
        result : FlowResults | dict
            The aerodynamic coefficients to be inserted, either as a PySAGAS native
            FlowResults object, or a dictionary with keys matching the data columns
            specified on instantiation of the deck.

        See Also
        --------
        FlowResults
        """
        # Check inputs
        inputs_given = [i in kwargs for i in self._inputs]
        if not all(inputs_given):
            raise Exception(
                "Please provide all input values when inserting new result."
            )

        # Process results
        if isinstance(result, FlowResults):
            # Get coefficients
            coefficients = result.coefficients(A_ref=self._a_ref, c_ref=self._c_ref)

            # Extract data
            # TODO - use columns provided in init
            data = {
                "CL": coefficients[0],
                "CD": coefficients[1],
                "Cm": coefficients[2],
            }

        else:
            # Check keys
            data_given = [i in result for i in self._columns]
            if not all(data_given):
                raise Exception(
                    "Please provide a data point for all values when inserting new result."
                )

            # Data provided directly
            data = result

        # Add inputs
        data.update(kwargs)

        # Add to deck
        self._deck = pd.concat(
            [self._deck, pd.DataFrame(data, index=[len(self._deck)])]
        )

    @classmethod
    def from_csv(
        cls, filepath: str, inputs: list[str], a_ref: float = 1, c_ref: float = 1
    ):
        """Create an Aerodeck from a CSV file.

        Parameters
        ----------
        filepath : str
            The filepath to the csv file containing aerodeck data.

        inputs : list[str]
            A list of the inputs to this aerodeck. For example, ["aoa", "mach"].

        a_ref : float, optional
            The reference area. The default is 1.

        c_ref : float, optional
            The reference length. The default is 1.
        """

        # Read data from file
        data = pd.read_csv(filepath)

        # Extract inputs and columns
        columns = list(data.columns)
        inputs = [columns.pop(columns.index(coef)) for coef in inputs]

        # Instantiate aerodeck
        aerodeck = cls(inputs=inputs, columns=columns, a_ref=a_ref, c_ref=c_ref)
        aerodeck._deck = data
        return aerodeck


class SensDeck(MultiDeck):
    TYPE = "sensdeck"

    def __init__(
        self,
        inputs: list[str],
        parameters: list[str],
        a_ref: float = 1,
        c_ref: float = 1,
        **kwargs,
    ) -> None:
        """Instantiate a new sensitivity deck.

        This object is a collection of `AeroDeck`s, containing aerodynamic sensitivity
        information.

        Parameters
        ----------
        inputs : list[str]
            A list of the inputs to this deck. For example, ["aoa", "mach"].

        parameters : list[str]
            A list of the parameters to this deck. For example, ["wingspan", "length"].

        a_ref : float, optional
            The reference area. The default is 1.

        c_ref : float, optional
            The reference length. The default is 1.

        See Also
        --------
        AeroDeck
        """

        # Save reference properties
        self._a_ref = a_ref
        self._c_ref = c_ref

        # Create base sensitivity deck
        columns = ["dCL", "dCD", "dCm"]
        base_deck = AeroDeck(inputs, columns, a_ref, c_ref)

        # Complete instantiation
        super().__init__(
            inputs=inputs, parameters=parameters, base_deck=base_deck, **kwargs
        )

    def __repr__(self) -> None:
        return f"{self.TYPE}"

    def __str__(self) -> None:
        return self.__repr__()

    def insert(self, result: SensitivityResults, **kwargs):
        # Get coefficients
        f_sens, m_sens = result.coefficients(A_ref=self._a_ref, c_ref=self._c_ref)

        for param in self._parameters:
            # Extract data for this parameter
            data = {
                "dCL": f_sens.loc[param]["dCL"],
                "dCD": f_sens.loc[param]["dCD"],
                "dCm": m_sens.loc[param]["dMz/dp"],
            }

            # Insert this data into the respective parameter deck
            self._decks[param].insert(result=data, **kwargs)

    @classmethod
    def from_csv(
        cls,
        param_filepaths: dict[str, str],
        inputs: list[str],
        a_ref: float = 1,
        c_ref: float = 1,
    ):
        """Create a SensDeck from a collection of CSV files.

        Parameters
        ----------
        param_filepaths : dict[str, str]
            A dictionary of filepaths, keyed by the associated parameter.

        inputs : list[str]
            A list of the inputs to this aerodeck. For example, ["aoa", "mach"].

        a_ref : float, optional
            The reference area. The default is 1.

        c_ref : float, optional
            The reference length. The default is 1.
        """
        decks = {}
        for param, filepath in param_filepaths.items():
            # Load data
            deck = AeroDeck.from_csv(
                filepath=filepath, inputs=inputs, a_ref=a_ref, c_ref=c_ref
            )
            decks[param] = deck

        # Extract inputs
        inputs = deck._inputs

        # Instantiate sensdeck
        sensdeck = cls(
            inputs=inputs, parameters=param_filepaths.keys(), a_ref=a_ref, c_ref=c_ref
        )

        # Overwrite with loaded decks
        sensdeck._decks = decks

        return sensdeck

    def to_csv(self, file_prefix: Optional[str] = None):
        """Save the decks to CSV file.

        Parameters
        ----------
        file_prefix : str, optional
            The CSV file name prefix. If None is provided, the deck __repr__
            will be used. The default is None.
        """
        file_prefix = file_prefix if file_prefix else self.__repr__()
        for p, deck in self._decks.items():
            deck.to_csv(file_prefix=f"{p}_{file_prefix}")

    @property
    def decks(self) -> dict[str, AeroDeck]:
        decks = {p: deck.deck for p, deck in self._decks.items()}
        return decks
