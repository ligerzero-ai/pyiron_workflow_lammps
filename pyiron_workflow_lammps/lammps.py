from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import os
import textwrap
import warnings
import logging

from ase import Atoms
import pyiron_workflow as pwf
from pyiron_workflow_lammps.generic import shell, create_WorkingDirectory
from pyiron_lammps import parse_lammps_output_files, write_lammps_structure


logger = logging.getLogger(__name__)

@dataclass
class LammpsInput:
    # If raw_script is set, it will override everything else:
    raw_script: Optional[str] = None

    units: str = "metal"
    dimension: int = 3
    boundary: Union[str, Tuple[str, str, str]] = ("p", "p", "p")
    atom_style: str = "atomic"
    read_data_file: str = "lammps.data"
    pair_style: str = "grace"
    pair_coeff: Tuple[str, str, str, str] = field(
        default_factory=lambda: ("*", "*", "/path/to/model", "Fe")
    )
    compute_id: str = "eng"
    dump_every: int = 10
    dump_filename: str = "dump.out"
    thermo_style_fields: Sequence[str] = field(
        default_factory=lambda: (
            "step", "temp", "pe", "etotal",
            "pxx", "pxy", "pxz", "pyy", "pyz", "pzz", "vol"
        )
    )
    thermo_format: str = "%20.15g"
    thermo_every: int = 10
    min_style: str = "cg"

    # Individual minimize parameters
    etol: float = 0.0        # energy tolerance
    ftol: float = 0.01       # force tolerance
    maxiter: int = 1_000_000 # maximum iterations
    maxeval: int = 1_000_000 # maximum force evaluations

    def generate(self) -> str:
        # If user provided a raw script, warn and return it directly:
        if self.raw_script is not None:
            warnings.warn(
                "Raw LAMMPS script provided; all other dataclass fields will be ignored.",
                UserWarning
            )
            return self.raw_script

        # normalize boundary
        b = self.boundary
        boundary_str = b if isinstance(b, str) else " ".join(b)

        lines = [
            f"units {self.units}",
            f"dimension {self.dimension}",
            f"boundary {boundary_str}",
            f"atom_style {self.atom_style}",
            "",
            f"read_data {self.read_data_file}",
            "",
            f"pair_style {self.pair_style}",
            f"pair_coeff {self.pair_coeff[0]} {self.pair_coeff[1]} "
            f"{self.pair_coeff[2]} {self.pair_coeff[3]}",
            "",
            "# per-atom potential energy",
            f"compute {self.compute_id} all pe/atom",
            "",
            f"# dump every {self.dump_every} steps: coords, forces, velocities, AND per-atom energy",
            f"dump 1 all custom {self.dump_every} {self.dump_filename} "
            f"id type xsu ysu zsu fx fy fz vx vy vz c_{self.compute_id}",
            (
                'dump_modify 1 sort id format line '
                '"%d %d %20.15g %20.15g %20.15g %20.15g '
                '%20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"'
            ),
            "",
            "# global thermo output to log",
            "thermo_style custom " + " ".join(self.thermo_style_fields),
            f"thermo_modify format float {self.thermo_format}",
            f"thermo {self.thermo_every}",
            "",
            f"min_style   {self.min_style}",
            (
                "minimize    "
                f"{self.etol} {self.ftol} "
                f"{self.maxiter} {self.maxeval}"
            ),
        ]
        return textwrap.dedent("\n".join(lines))

    def __str__(self) -> str:
        return self.generate()
    
@pwf.as_function_node("working_directory")
def write_LammpsStructure(structure,
                         working_directory,
                         potential_elements,
                         units="metal",
                         file_name="lammps.data"):
    write_lammps_structure(
                            structure=structure,
                            potential_elements=potential_elements,
                            units=units,
                            file_name=file_name,
                            working_directory=working_directory,
                        )
    return working_directory

@pwf.as_function_node("filepath")
def write_LammpsInput(
    lmp_input: LammpsInput,
    filename: str,
    working_directory: str | None = None
) -> str:
    """
    Write a LAMMPS input script to disk and return the full file path.

    Parameters
    ----------
    lmp_input
        An instance of LammpsInput (your dataclass) containing all settings.
        If `lmp_input.raw_script` is set, that script is written verbatim.
    filename
        Name of the file to write (e.g. "in.lammps").
    working_directory
        Directory in which to place the file. If None, writes to the current
        working directory. The directory will be created if it doesn't exist.

    Returns
    -------
    filepath : str
        The full path to the file that was written.
    """
    script = str(lmp_input)

    if working_directory is not None:
        os.makedirs(working_directory, exist_ok=True)
        path = os.path.join(working_directory, filename)
    else:
        path = filename

    with open(path, "w") as f:
        f.write(script)

    return path

@pwf.as_function_node("lammps_output")
def parse_LammpsOutput(
    working_directory: str = os.getcwd(),
    structure: Atoms | None = None,
    potential_elements: List[str] | None = None,
    units: str = "metal",
    function: Callable[..., Any] | None = None,
    parser_args: dict[str, Any] = {}
) -> Any:
    # Filter out the LAMMPS unit conversion warning
    warnings.filterwarnings("ignore", message=".*Couldn't determine the LAMMPS to pyiron unit conversion type of quantity eng.*")
    """
    Parse LAMMPS output files in the working directory.

    Args:
        working_directory (str): Path to the directory containing LAMMPS outputs.
        structure (ase.Atoms, optional): The input structure used in the run.
        potential_elements (list[str], optional): List of element symbols in the potential.
        units (str, optional): Units style used in the run (default "metal").
        function (callable, optional): Custom parser function. Must accept the same
            keyword arguments as parse_lammps_output_files.
        parser_args (dict, optional): If `function` is provided, these args will
            be passed to it. Ignored if `function` is None.

    Returns:
        Whatever the parser returns (typically a dict of output data).
    """
    if function is None:
        # use the built-in parser
        from pyiron_lammps import parse_lammps_output_files
        # print(f"workdir in {working_directory}")
        parse_fn = parse_lammps_output_files
        parser_args = {
            "working_directory": working_directory,
            "structure": structure,
            "potential_elements": potential_elements,
            "units": units,
            "dump_h5_file_name": "dump.h5",
            "dump_out_file_name": "dump.out",
            "log_lammps_file_name": "minimize.log",
        }
    else:
        parse_fn = function

    return parse_fn(**parser_args)

def get_structure_species_lists(lammps_data_filepath = "lammps.data",
                               lammps_dump_filepath = "dump.out"):
    """
    Prompt for a LAMMPS data file path, parse its Masses section,
    and return a dict mapping atom-type indices to element symbols.
    """
    species_map = get_species_map(lammps_data_filepath)
    from pymatgen.io.lammps.outputs import parse_lammps_dumps
    species_lists = []
    for dump in parse_lammps_dumps(lammps_dump_filepath):
        species_lists.append([species_map[idx] for idx in dump.data.type])
    return species_lists

def get_species_map(lammps_data_filepath = "lammps.data"):
    species_map = {}
    with open(lammps_data_filepath, "r") as f:
        # find the "Masses" section
        for line in f:
            if line.strip() == "Masses":
                # skip the blank line immediately after
                next(f)
                break
        # read until the next blank line
        for line in f:
            stripped = line.strip()
            if not stripped:
                # end of the Masses block
                break
            # split off any comment after '#'
            parts = stripped.split("#", 1)
            # first token is "<index> <mass>"
            idx = int(parts[0].split()[0])
            # if there’s a comment like "(Fe)", strip parentheses
            symbol = parts[1].strip().strip("()") if len(parts) > 1 else None
            species_map[idx] = symbol
    return species_map

@pwf.as_macro_node("lammps_output")
def lammps_job(
    self,
    working_directory: str,
    structure: Atoms,
    lmp_input: LammpsInput,
    potential_elements: List[str],
    input_filename: str = "in.lmp",
    command: str = "mpirun -np 40 --bind-to none /cmmc/ptmp/hmai/LAMMPS/lammps_grace/build/lmp -in in.lmp -log minimize.log",
    lammps_parser_function: Callable[..., Any] | None = None,
    lammps_parser_args: dict[str, Any] = {},
):
    """
    Run a LAMMPS calculation end-to-end:
      1) create working_directory  
      2) write structure → data file  
      3) write LammpsInput → input script  
      4) shell out to LAMMPS  
      5) parse results with an (optional) custom parser  

    Args:
        working_directory: Path to the working directory.
        structure: ASE Atoms object to simulate.
        lmp_input: LammpsInput dataclass with all run settings.
        potential_elements: List of element symbols in the potential.
        input_filename: Name of the input script (default "in.lmp").
        command: Full MPI/launcher command to run LAMMPS.
        lammps_parser_function: If provided, use this instead of the default parse_lammps_output.
        lammps_parser_args: Dict of kwargs to pass to the parser function.

    Returns:
        Parsed LAMMPS output (whatever the parser returns).
    """
    # 1) Create the directory
    self.working_dir = create_WorkingDirectory(working_directory=working_directory)

    # 2) Write the ASE structure to LAMMPS data
    self.structure_writer = write_LammpsStructure(
        structure=structure,
        potential_elements=potential_elements,
        units=lmp_input.units,
        file_name=lmp_input.read_data_file,
        working_directory=working_directory
    )

    # 3) Write the input script
    self.input_writer = write_LammpsInput(
        lmp_input=lmp_input,
        filename=input_filename,
        working_directory=working_directory
    )

    # 4) Run LAMMPS
    self.job = shell(command=command,
                     working_directory=working_directory)

    # 5) Parse output (using custom or default parser)
    self.lammps_output = parse_LammpsOutput(
        working_directory=working_directory,
        structure=structure,
        potential_elements=potential_elements,
        units=lmp_input.units,
        function=lammps_parser_function,
        parser_args=lammps_parser_args
    )

    # Wire up the flow
    (
        self.working_dir
        >> self.structure_writer
        >> self.input_writer
        >> self.job
        >> self.lammps_output
    )

    # Register start
    self.starting_nodes = [self.working_dir]

    return self.lammps_output