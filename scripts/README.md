# Traffic Simulation Scripts

This folder contains scripts for running traffic simulations using the `TrafficModel` class. The scripts allow users to configure simulation parameters, automate batch simulations, and test different optimization techniques.

## Files in This Folder

### 1. `sim.py`

The main script for running a traffic simulation.

- **Description**: Simulates traffic patterns based on user-defined parameters such as the number of intersections, cars, borders, and optimization type.
- **Key Features**:
  - Configurable via command-line arguments.
  - Collects and stores simulation data in `.parquet` format.
  - Outputs a JSON file with the simulation configuration.
- **Usage**:

  ```bash
  # Run from the project's root directory
  python scripts/sim.py -i 100 -c 500 -o advanced_ml -s 2000
  ```

  *(Note: Default values are used if arguments are omitted. See `sim.py --help` for details.)*

### 2. `sim_configs.sh`

A bash script to run multiple simulations with varying configurations.

- **Description**: Automates running `sim.py` with different combinations of `num_intersections`, `num_borders`, and `num_cars`. Useful for generating data across various simulation setups.
- **Key Features**:
  - Iterates through predefined arrays of simulation parameters.
  - Executes `sim.py` for each parameter combination.
- **Usage**:

  ```bash
  # Run from the project's root directory
  bash scripts/sim_configs.sh
  ```
  
  *(Note: Modify the parameter arrays within the script to change the configurations tested.)*

### 3. `sim_opt_types.sh`

A bash script to run simulations comparing different optimization types.

- **Description**: Automates running `sim.py` with a fixed configuration but iterates through different `optimization_type` options (`none`, `simple`, `advanced`, `advanced_ml`). Useful for comparing the impact of different traffic light control strategies.
- **Key Features**:
  - Iterates through a predefined list of optimization types.
  - Executes `sim.py` for each optimization type using otherwise consistent parameters.
- **Usage**:

  ```bash
  # Run from the project's root directory
  bash scripts/sim_opt_types.sh
  ```

  *(Note: Modify the fixed parameters or the `OPTIMIZATION_TYPES` array within the script as needed.)*

## General Instructions

- **Execution Context**: All scripts should be executed from the root directory of the `ias_project`.
- **Data Storage**: Each simulation run creates a timestamped subdirectory within the `data/` folder (located in the project root) to store the `.parquet` data files and the `config.json`.
- **Dependencies**: Ensure all necessary Python packages listed in the project's requirements are installed. The simulation relies on the `TrafficModel` from the `src/` directory.
