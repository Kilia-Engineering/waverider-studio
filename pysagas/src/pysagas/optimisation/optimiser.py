import os
import numpy as np
from pysagas import banner
from numpy.typing import ArrayLike
from typing import Union, Optional, Dict, Any
from pyoptsparse import Optimizer, Optimization


class ShapeOpt:
    """PySAGAS Shape Optimisation wrapper for pyOptSparse."""

    def __init__(
        self,
        optimiser: Optimizer,
        working_dir: str,
        optimiser_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise PySAGAS Shape Optimiser.

        Parameters
        ----------
        optimiser : Optimizer
            The pyoptsparse Optimizer object of choice.

        working_dir : str
            The name of the working directory.

        optimiser_options : dict, optional
            The options to pass to the optimiser.
        """
        self.opt_problem: Optimization
        self.optimiser = optimiser(options=optimiser_options)

        # Prepare working directory
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

    def add_variables(
        self,
        parameters_dict: Optional[dict] = None,
        name: Optional[str] = None,
        n: Optional[int] = None,
        vartype: Optional[str] = "c",
        initial: Optional[ArrayLike] = None,
        lower_bounds: Optional[Union[ArrayLike, Dict]] = None,
        upper_bounds: Optional[Union[ArrayLike, Dict]] = None,
        **kwargs,
    ):
        """Adds variables to the optimisation problem.

        Parameters
        ----------
        parameters_dict : dict, optional
            The dictionary of nominal parameter values. This can be provided
            instead of adding each parameter individually with the 'name'
            argument.

        name : str
            Name of variable group. This name should be unique across all the design variable groups

        n : int, optional
            Number of design variables in this variable group.

        varType : str.
            String representing the type of variable. Suitable values for type
            are: 'c' for continuous variables, 'i' for integer values and
            'd' for discrete selection.

        value : scalar or array.
            Starting value for design variables. If it is a a scalar, the same
            value is applied to all 'nVars' variables. Otherwise, it must be
            iterable object with length equal to 'nVars'.

        lower : scalar or array.
            Lower bound of variables. Scalar/array usage is the same as value
            keyword

        upper : scalar or array.
            Upper bound of variables. Scalar/array usage is the same as value
            keyword
        """

        def check_bounds(
            parameters: Dict[str, float], bounds: Dict[str, float], sign: int
        ):
            for p, v in parameters.items():
                # Get bound for this parameter
                b = bounds.get(p)
                if np.sign(v - b) != np.sign(sign) and np.sign(v - b) != 0:
                    if np.sign(sign) < 0:
                        bound = ">"
                    else:
                        bound = "<"
                    raise Exception(
                        f"Invalid bounds on {p}: " + f"{v:.5f} {bound} {b:.5f}."
                    )

        if parameters_dict:
            # Check bounds
            if lower_bounds is None:
                lower_bounds = {}
            else:
                # Make sure bounds are valid
                check_bounds(parameters_dict, lower_bounds, 1)

            if upper_bounds is None:
                upper_bounds = {}
            else:
                # Make sure bounds are valid
                check_bounds(parameters_dict, upper_bounds, -1)

            # Unpack parameters dict
            for param, value in parameters_dict.items():
                self.opt_problem.addVarGroup(
                    name=param,
                    nVars=1,
                    varType=vartype,
                    value=value,
                    lower=lower_bounds.get(param),
                    upper=upper_bounds.get(param),
                    **kwargs,
                )

        else:
            self.opt_problem.addVarGroup(
                name=name,
                nVars=n,
                varType=vartype,
                value=initial,
                lower=lower_bounds,
                upper=upper_bounds,
                **kwargs,
            )

    def add_constriants(
        self,
        name: str,
        n: Optional[int] = 1,
        lower: Optional[Union[ArrayLike, Dict]] = None,
        upper: Optional[Union[ArrayLike, Dict]] = None,
        scale: Optional[float] = 1,
    ):
        """Adds constraints to the optimisation problem.

        Parameters
        -----------
        name : str
            The name key of the constraint being added. This must appear
            in the dictionary returned by the objective and Jacobian callback
            returns.

        nCon : int
            The number of constraints in this group

        lower : scalar or array
            The lower bound(s) for the constraint. If it is a scalar,
            it is applied to all nCon constraints. If it is an array,
            the array must be the same length as nCon.

        upper : scalar or array
            The upper bound(s) for the constraint. If it is a scalar,
            it is applied to all nCon constraints. If it is an array,
            the array must be the same length as nCon.

        scale : scalar or array
            A scaling factor for the constraint. It is generally
            advisable to have most optimization constraint around the
            same order of magnitude.
        """
        self.opt_problem.addConGroup(
            name=name,
            nCon=n,
            lower=lower,
            upper=upper,
            scale=scale,
            linear=None,
            wrt=None,
            jac=None,
        )

    def add_objective(self, name: str):
        """Adds an objective to the optimisation problem.

        Parameters
        -----------
        name : str
            The name key of the objective being added. This must appear
            in the dictionary returned by the objective and Jacobian callback
            returns.
        """
        self.opt_problem.addObj(name)

    def run(self, hotstart_file: str = None):
        """Run ShapeOpt.

        Parameters
        -----------
        hotstart_file : str, optional
            The filepath to the history file, used to hot start the
            optimiser.
        """
        # Print banner
        banner()
        print("\033[4mPySAGAS Shape Optimisation\033[0m".center(50, " "))

        # Run optimiser
        self.sol = self.optimiser(
            self.opt_problem,
            storeHistory="history.hst",
            hotStart=hotstart_file,
        )
