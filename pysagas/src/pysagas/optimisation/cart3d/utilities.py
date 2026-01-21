import os
import time
import shutil
from random import random
from typing import Optional
from hypervehicle.utilities import append_sensitivities_to_tri


class C3DPrep:
    def __init__(
        self,
        logfile,
        jitter_denom: float = 1000,
        rotation_attempts: int = 6,
        info_file: str = None,
        write_config: bool = True,
    ) -> None:
        self._logfile = logfile
        self._info = info_file if info_file is not None else self._logfile
        self._jitter_denom = jitter_denom  # for 1000; Max of 0.0001, min of 0
        self._rotation_attempts = rotation_attempts
        self._write_config = write_config

    def _run_stl2tri(self, stl_files: list):
        tri_files = []
        for file in stl_files:
            prefix = file.split(".")[0]
            tri_file = prefix + ".tri"
            os.system(f"stl2tri.pl {file} {tri_file} >> {self._logfile} 2>&1")
            tri_files.append(tri_file)

        os.system(f"rm *.comp.tri *.off >> {self._logfile} 2>&1")
        return tri_files

    def _jitter_tri_files(self, tri_files):
        for file in tri_files:
            prefix = file.split(".")[0]
            x_pert = random() / self._jitter_denom
            y_pert = random() / self._jitter_denom
            z_pert = random() / self._jitter_denom
            os.system(
                f"trix -x {x_pert} -y {y_pert} -z {z_pert} -o {prefix} {file} >> {self._logfile} 2>&1"
            )

    def _shift_all(
        self,
        tri_files,
        x_shift,
        y_shift,
        z_shift,
        component: str = None,
        reverse: bool = False,
    ):
        if component is None:
            transform_files = tri_files
        else:
            transform_files = [component]

        for file in transform_files:
            prefix = ".".join(file.split(".")[:-1])
            if reverse:
                os.system(
                    f"trix -x {-x_shift} -y {-y_shift} -z {-z_shift} -o {prefix} {file} >> {self._logfile} 2>&1"
                )
            else:
                os.system(
                    f"trix -x {x_shift} -y {y_shift} -z {z_shift} -o {prefix} {file} >> {self._logfile} 2>&1"
                )

    def _rotate_all(
        self,
        tri_files,
        x_rot,
        y_rot,
        z_rot,
        component: str = None,
        reverse: bool = False,
    ):
        # Define rotations dict
        rotations = {
            "x": x_rot,
            "y": y_rot,
            "z": z_rot,
        }

        # Determine files to be transformed
        if component is None:
            # Rotate all .tri files (as provided)
            transform_files = tri_files
        else:
            # Rotate the single component
            transform_files = [component]

        for file in transform_files:
            prefix = ".".join(file.split(".")[:-1])

            # Check order of operations
            if reverse:
                order = ["z", "y", "x"]
            else:
                order = ["x", "y", "z"]

            # Apply rotations
            for axis in order:
                rotation = rotations[axis]
                os.system(
                    f"trix -r{axis} {rotation} -o {prefix} {file} >> {self._logfile} 2>&1"
                )

    def _run_comp2tri(self, tri_files):
        tri_files_str = " ".join(tri_files)

        if os.path.exists("Config.xml"):
            # Remove old config file
            os.remove("Config.xml")

        # Run command
        conf_str = "-config" if self._write_config else ""
        os.system(
            f"comp2tri -makeGMPtags {tri_files_str} {conf_str} >> {self._logfile} 2>&1"
        )

        if self._write_config:
            # Await Config.xml
            while not os.path.exists("Config.xml"):
                time.sleep(0.2)

            # Overwrite Config.xml Component names using tri files
            original = os.path.join("Config.xml")
            new = os.path.join("temp_config.xml")
            with open(new, "w+") as new_file:
                with open(original, "r") as original_file:
                    for line in original_file:
                        if "Component Name" in line:
                            # Get component number
                            name_prefix = "Component_"
                            name_start = line.index(name_prefix)
                            comp_no = line.split('"')[1].split("_")[-1]
                            tri_prefix = tri_files[int(comp_no) - 1].split(".")[0]

                            # Update line
                            line = (
                                line[:name_start]
                                + tri_prefix
                                + line[name_start + len(name_prefix) + 1 :]
                            )

                        # Write line to new file
                        new_file.write(line)

            # # Replace original aero.csh file with updated file
            shutil.copymode(original, new)
            os.remove(original)
            os.rename(new, original)

        else:
            # Await Components.tri
            while not os.path.exists("Components.tri"):
                time.sleep(0.2)

    def _run_intersect(self):
        os.system(f"intersect >> {self._logfile} 2>&1")

    @staticmethod
    def _get_stl_files() -> list[str]:
        path = os.getcwd()
        all_files = [
            f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
        ]
        stl_files = []
        for file in all_files:
            if file.split(".")[-1] == "stl":
                stl_files.append(file)

        # Sort files
        stl_files.sort()

        return stl_files

    @staticmethod
    def _check_for_success():
        success = False
        if os.path.isfile("Components.i.tri"):
            success = True
        return success

    def intersect_stls(self, stl_files: Optional[list[str]] = None) -> bool:
        """Create Components.i.tri by intersecting all STL files."""
        # Check for existing intersected file
        if self._check_for_success():
            self._log("Intersected components file already present.")
            return True

        # Continue
        if not stl_files:
            stl_files = self._get_stl_files()
        tri_files = self._run_stl2tri(stl_files)

        # First try intersect original files
        self._log("Making first attempt to intersect components with no perturbations.")
        self._run_comp2tri(tri_files)
        self._run_intersect()
        successful = self._check_for_success()
        if successful:
            self._log("Success.")
            return True

        # That failed, try jittering components
        self._log("Attempt failed, now attempting to intersect jittered components.")
        self._jitter_tri_files(tri_files)
        self._run_comp2tri(tri_files)
        self._run_intersect()
        successful = self._check_for_success()
        if successful:
            self._log("Success.")
            return True

        # That failed, try arbitrary shifts away
        self._log("Attempt failed, now attempting random perturbations.")
        for attempt in range(self._rotation_attempts):
            # Define shifts
            x_shift = random() * 10  # Max of 10, min of 0
            y_shift = random() * 10  # Max of 10, min of 0
            z_shift = random() * 10  # Max of 10, min of 0

            # Define rotations
            x_rot = random() * 10  # Max of 10, min of 0
            y_rot = random() * 10  # Max of 10, min of 0
            z_rot = random() * 10  # Max of 10, min of 0

            # Apply transformations
            self._shift_all(
                tri_files=tri_files, x_shift=x_shift, y_shift=y_shift, z_shift=z_shift
            )
            self._rotate_all(tri_files=tri_files, x_rot=x_rot, y_rot=y_rot, z_rot=z_rot)

            if attempt > 0:
                # Also jitter
                self._log(f"On attempt {attempt+1}, also applying random jitter.")
                self._jitter_tri_files(tri_files)

            # Make intersect attempt
            self._run_comp2tri(tri_files)
            self._run_intersect()
            successful = self._check_for_success()

            if successful:
                # Move configuration back to original location
                self._log(
                    "Success. Now moving Components.i.tri back to original position."
                )
                self._rotate_all(
                    tri_files=tri_files,
                    x_rot=-x_rot,
                    y_rot=-y_rot,
                    z_rot=-z_rot,
                    component="Components.i.tri",
                    reverse=True,  # Perform in reverse order
                )
                self._shift_all(
                    tri_files=tri_files,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    z_shift=z_shift,
                    component="Components.i.tri",
                    reverse=True,
                )
                return True

            else:
                # Need to reset tri files
                self._log(
                    f"Random perturbation attempt {attempt+1} failed. Resetting "
                    + ".tri files and attempting again."
                )
                tri_files = self._run_stl2tri(stl_files)

        # Finish log
        self._log("Unsuccessful.")

        return False

    def run_autoinputs(self, r: int = 2, *args):
        """Runs autoInputs to create input.c3d.

        Parameters
        -----------
        r : int, optional
            The distance to farfield normalized by max geometry size. The default is 2.
        args : str, optional
            Any extra arguments to provide to autoInputs, as strings. For arguments with
            values, just include the value in the string.
        """
        extra_args = "-" + " -".join(args) if args else ""
        os.system(f"autoInputs -r 2 {extra_args} >> {self._logfile} 2>&1")

    def run_aero(self):
        """Runs aero.csh."""
        os.system("./aero.csh restart")

    def _log(self, msg: str):
        with open(self._info, "a") as f:
            f.write("\nC3DPrep: " + msg)


def combine_sense_data(
    components_filepath: str,
    sensitivity_files: list[str],
    match_target: float = 0.9,
    tol_0: float = 1e-5,
    max_tol: float = 1e-1,
    outdir: Optional[str] = None,
    verbosity: Optional[int] = 1,
):
    """Combine the individual component sensitivity data with the
    intersected geometry file (eg. Components.i.tri).
    """
    match_frac = 0
    tol = tol_0
    while match_frac < match_target:
        # Check tolerance
        if tol > max_tol:
            raise Exception(
                "Cannot combine sensitivity data (match fraction: "
                + f"{match_frac}, tolerance: {tol}, max tolerance: {max_tol})."
            )

        # Run matching algorithm
        match_frac = append_sensitivities_to_tri(
            dp_filenames=sensitivity_files,
            components_filepath=components_filepath,
            match_tolerance=tol,
            verbosity=verbosity,
            outdir=outdir,
        )

        if match_frac < match_target and verbosity > 0:
            print(
                "Failed to combine sensitivity data "
                f"({100*match_frac:.02f}% match rate)."
            )
            print(f"  Increasing matching tolerance to {tol*10} and trying again.")
        elif verbosity > 0:
            print(
                "Component sensitivity data matched to intersected geometry "
                + f"with {100*match_frac:.02f}% match rate."
            )

        # Increase matching tolerance
        tol *= 10
