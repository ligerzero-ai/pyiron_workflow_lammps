# pyiron_workflow_lammps

A LAMMPS workflow integration package for pyiron, providing tools and utilities for running and managing LAMMPS calculations within the pyiron workflow framework.

## Installation

The package can be installed using pip:

```bash
# Basic installation
pip install pyiron_workflow_lammps

# Installation with development tools
pip install "pyiron_workflow_lammps[dev]"

# Installation with notebook support
pip install "pyiron_workflow_lammps[notebook]"

# Installation with all optional dependencies
pip install "pyiron_workflow_lammps[dev,notebook]"
```

## Dependencies

The package requires the following core dependencies:
- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- pyiron_workflow >= 0.1.0
- ase >= 3.22.0
- pyiron_lammps

Optional dependencies for development:
- pytest >= 6.0
- pytest-cov >= 2.0
- black >= 22.0
- isort >= 5.0
- flake8 >= 4.0

Optional dependencies for notebook support:
- jupyter >= 1.0.0
- notebook >= 6.0.0

## Project Information

- **License**: BSD-3-Clause
- **Development Status**: Alpha
- **Documentation**: [GitHub Repository](https://github.com/pyiron/pyiron_workflow_lammps)
- **Bug Tracker**: [GitHub Issues](https://github.com/pyiron/pyiron_workflow_lammps/issues)

## Usage

The package provides a set of tools for running LAMMPS calculations within pyiron workflows. Here's a basic example:

```python
from pyiron_workflow_lammps import lammps

# Create a LAMMPS job
job = lammps.lammps_job(
    structure=your_structure,
    input_parameters={
        "units": "metal",
        "timestep": 0.001,
        "thermo_style": "custom step temp pe ke etotal press vol"
    }
)

# Run the job
job.run()
```

## Example Notebooks

The package includes example notebooks to help you get started:

- `QuickStart.ipynb`: A comprehensive guide to using the package
- `test_lammps.ipynb`: Examples of running LAMMPS calculations

You can find these notebooks in the `example_notebooks` directory.

## LAMMPS Configuration

To use LAMMPS with pyiron_workflow, you need to ensure that LAMMPS is properly installed and accessible in your system. The package will look for the LAMMPS executable in your system PATH or in a specified location.

### Setting up LAMMPS

1. Install LAMMPS on your system following the official installation guide from [LAMMPS website](https://docs.lammps.org/Install.html)

2. Ensure the LAMMPS executable is in your system PATH or specify its location in your pyiron configuration

3. Verify your LAMMPS installation by running:
```bash
lmp -version
```

### Additional Notes:
- Make sure you have the necessary LAMMPS packages installed for your specific simulation needs
- The package supports both serial and parallel LAMMPS execution
- For optimal performance, consider using MPI-enabled LAMMPS builds for parallel calculations
- The package integrates with pyiron's job management system for efficient workflow handling
