# Nonlinear Lifting Line

Nonlinear lifting line aerodynamic analysis code

## Installation

1. Clone this repo
2. (Optional) Create a [virtual environment](https://docs.python.org/3/library/venv.html)
3. Run `pip install -e .` -- this installs the package as well as its dependencies automatically.

## Usage

The `nll` package contains several modules which can be used for aero calculations. An example is shown in the script `scripts/main.py`.

### Calling `main.py`

Accepts one argument for changing the aero data

```plaintext
usage: main.py [-h] [--data_dir DATA_DIR]

Calculate lift and drag coefficients for a finite wing

options:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Path to folder containing aero data,
                       default n0012_xfoil_data
  ```

### Running Tests

Run `pip install -e .[dev]` to install dependencies for development (currently just `pytest`).

Run unit tests using the command `pytest` from the root of the repo.
