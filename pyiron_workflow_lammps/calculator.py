from pyiron_workflow_lammps.lammps import lammps_job
import pyiron_workflow as pwf
from ase import Atoms
from .lammps import LammpsInput
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from ase import Atoms
import numpy as np
from pyiron_workflow_lammps.lammps import get_species_map
import os


def arrays_to_ase_atoms(
    cells:      np.ndarray,        # (n_frames, 3, 3)
    positions:  np.ndarray,        # (n_frames, n_atoms, 3)
    indices:    np.ndarray,        # (n_frames, n_atoms)
    species_lists: list[list[str]],   # e.g. {0:'Fe',1:'C'} or {1:'Fe',2:'C'}
    pbc:        bool = True,
) -> Atoms:
    """
    Convert the final frame of LAMMPS‐style arrays into an ASE Atoms.

    Raises KeyError if any type in `indices` isn't a key in `species_map`.
    """
    # last frame
    cell  = cells
    pos   = positions
    types = indices

    return Atoms(symbols=species_lists[-1], positions=pos, cell=cell, pbc=pbc)


@pwf.as_function_node("atoms", "final_results", "converged")
def get_calculator_outputs_from_lammps_node_output(lammps_node_output,
                                                   working_directory,
                                                   species_lists,
                                                   lammps_log_filepath = "minimize.log",
                                                   convergence_printout = "Total wall time:"):
    final_results = {}
    # print(lammps_node_output["generic"]["indices"], len(lammps_node_output["generic"]["indices"]))
    final_results["energy"] = lammps_node_output["generic"]["energy_tot"][-1]
    final_results["forces"] = lammps_node_output["generic"]["forces"][-1]
    final_results["stresses"] = lammps_node_output["generic"]["pressures"][-1]
    
    # print(lammps_node_output["generic"]["cells"], len(lammps_node_output["generic"]["cells"]))
    atoms = arrays_to_ase_atoms(
                                cells = lammps_node_output["generic"]["cells"][-1],
                                positions = lammps_node_output["generic"]["positions"][-1],
                                indices = lammps_node_output["generic"]["indices"][-1],
                                species_lists = species_lists
                            )
    final_results["cell"] = atoms.get_cell().tolist()
    final_results["volume"] = atoms.get_volume()
    from pyiron_workflow_lammps.generic import isLineInFile
    import os
    if isLineInFile.node_function(filepath = os.path.join(working_directory, lammps_log_filepath),
                                  line = convergence_printout,
                                  exact_match = False):
        # print(os.path.join(working_directory, lammps_log_filepath), convergence_printout)
        converged = True
    else:
        converged = False
    return atoms, final_results, converged

#@pwf.as_macro_node("atoms", "final_results", "converged")
def lammps_calculator_node(working_directory: str,
                            structure: Atoms,
                            lmp_input: LammpsInput,
                            potential_elements: List[str],
                            input_filename: str = "in.lmp",
                            command: str = "mpirun -np 40 --bind-to none /cmmc/ptmp/hmai/LAMMPS/lammps_grace/build/lmp -in in.lmp -log minimize.log",
                            lammps_log_filepath: str = "minimize.log",
                            lammps_log_convergence_printout: str = "Total wall time:",
                            # Don't offer the parsing interface because we must use the pyiron_lammps parser for knowledge on how to parse the energies/forces/stresses
                            # lammps_parser_function: Callable[..., Any] | None = None,
                            # lammps_parser_args: dict[str, Any] = {},
                           ):
    
    lammps_output = lammps_job(working_directory = working_directory,
                                structure = structure,
                                lmp_input = lmp_input,
                                input_filename = input_filename,
                                potential_elements = potential_elements,
                                command = command).run()
    from pyiron_workflow_lammps.lammps import get_structure_species_lists
    species_lists = get_structure_species_lists(lammps_data_filepath = os.path.join(working_directory, lmp_input.read_data_file),
                                                             lammps_dump_filepath = os.path.join(working_directory, lmp_input.dump_filename))
    # np.unique(structure.get_chemical_symbols())
    lammps_output = lammps_output["lammps_output"]
    atoms, final_results, converged = get_calculator_outputs_from_lammps_node_output.node_function(lammps_output,
                                                                               working_directory = working_directory,
                                                                               species_lists = species_lists,
                                                                               lammps_log_filepath = lammps_log_filepath,
                                                                               convergence_printout = lammps_log_convergence_printout)
    return atoms, final_results, converged

def lammps_engine_node(working_directory: str,
                            structure: Atoms,
                            lammps_engine: LammpsEngine,
                            
                            # Don't offer the parsing interface because we must use the pyiron_lammps parser for knowledge on how to parse the energies/forces/stresses
                            # lammps_parser_function: Callable[..., Any] | None = None,
                            # lammps_parser_args: dict[str, Any] = {},
                           ):
    
    lammps_output = lammps_job(working_directory = working_directory,
                                structure = structure,
                                lmp_input = lmp_input,
                                input_filename = input_filename,
                                potential_elements = potential_elements,
                                command = command).run()
    from pyiron_workflow_lammps.lammps import get_structure_species_lists
    species_lists = get_structure_species_lists(lammps_data_filepath = os.path.join(working_directory, lmp_input.read_data_file),
                                                             lammps_dump_filepath = os.path.join(working_directory, lmp_input.dump_filename))
    # np.unique(structure.get_chemical_symbols())
    lammps_output = lammps_output["lammps_output"]
    atoms, final_results, converged = get_calculator_outputs_from_lammps_node_output.node_function(lammps_output,
                                                                               working_directory = working_directory,
                                                                               species_lists = species_lists,
                                                                               lammps_log_filepath = lammps_log_filepath,
                                                                               convergence_printout = lammps_log_convergence_printout)
    return atoms, final_results, converged