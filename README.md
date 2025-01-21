# py-mlmc

`py-mlmc` is a Python project dedicated to the implementation and testing of **multi-level Monte Carlo (MLMC)** methods and **finite difference methods (FDM)** for numerical simulations in finance. The project explores their applications, performance, and accuracy in solving financial problems.

## Installation

To use this project, make sure you have Python installed and the required libraries. Install the dependencies using :
```sh
pip install numpy matplotlib scipy
```

## Usage

### Running Core Implementations

The `py_mlmc` directory contains the core Python files. To run them, simply import the desired modules into your scripts or notebooks.

### Notebooks

Navigate to the `notebooks` directory to explore the testing and validation notebooks:
- [`tests_fdm.ipynb`](notebooks/tests_fdm.ipynb): Demonstrates and validates the FDM implementation.
- [`tests_mlmc.ipynb`](notebooks/tests_mlmc.ipynb): Demonstrates and validates the MLMC implementation.

Launch the notebooks using :
```sh
jupyter notebook notebooks/tests_fdm.ipynb
```

## Documentation

- **Final Report**: The [report.pdf](docs/report.pdf) file provides detailed documentation of the project, including :
    - Implementation details.
    - Analysis of results.
- **Research Paper**: [giles.mlmc.pdf](docs/giles.mlmc.pdf) is the reference research paper used to guide the MLMC implementation.

## Authors

This project was collaboratively developed by :
- Paul-Antoine Leveilley
- Ryan Timeus

We welcome feedback and suggestions for improvements to this project.
