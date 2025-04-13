# DRIVE: Data-driven Road Intelligence for Vehicular Environments

This project implements an agent-based traffic simulation using the Mesa framework. It models vehicle movement (`CarAgent`) on a graph-based road network (`Graph`) with traffic lights (`LightAgent`) at intersections. The simulation includes visualization via Streamlit and Plotly, various traffic light optimization strategies (including ML-based prediction), and scripts for running batch simulations and training the machine learning model.

## Features

* **Agent-Based Simulation:** Models individual cars and traffic lights as agents using the Mesa framework.
* **Graph-Based Network:** Represents the road network using NetworkX, allowing for complex city layouts.
* **Interactive Visualization:** A Streamlit application (`app.py`) provides a real-time visualization of the simulation using Plotly (`graph_viz.py`).
* **Traffic Light Optimization:** Implements multiple strategies for traffic light control:
  * `none`: No optimization (fixed cycle or random).
    * `simple`: Basic optimization based on immediate waiting cars.
    * `advanced`: Optimization using linear programming (requires Gurobi/HiGHS).
    * `advanced_ml`: Optimization incorporating machine learning predictions of future arrivals (requires Gurobi/HiGHS and a trained model).
* **Machine Learning Integration:** Includes notebooks and scripts (`ml_training/`) to train a regression model (XGBoost or Neural Network) to predict traffic flow, which can be used by the `advanced_ml` optimizer.
* **Data Collection:** Uses Polars for efficient collection and storage of simulation data (car movements, wait times, light states) in Parquet format.
* **Scriptable Simulations:** Provides Python scripts (`scripts/`) and bash helpers to run simulations with varying configurations and collect data systematically.
* **Docker Support:** Includes `Dockerfile` and `docker-compose.yaml` for containerized execution of the Streamlit app and tests.

## Project Structure

```text
ias_project/
├── .secrets/
│   └── gurobi.lic        # REQUIRED: Gurobi license file (if using advanced optimizers)
├── artifacts/
│   ├── model.pkl         # Trained ML model artifact
│   └── scaler.pkl        # Scaler artifact for ML model
├── data/                 # Stores output data from simulation runs
├── docs/                 # Project documentation (e.g., reports)
├── ml_training/          # Jupyter notebooks and scripts for ML model training
│   ├── mlruns/           # MLflow tracking data
│   └── README.md         # ML training
├── scripts/              # Scripts for running batch simulations
│   └── README.md         # Scripts
├── src/                  # Core source code for the simulation model and agents
│   └── README.md         # Source code
├── tests/                # Unit tests
├── app.py                # Main Streamlit application file
├── Dockerfile            # Docker configuration for the application
├── docker-compose.yaml   # Docker Compose configuration
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup

### Prerequisites

* Python 3.12
* Docker and Docker Compose (Optional, for containerized execution)
* Gurobi Solver & License (Required *only* if using `advanced` or `advanced_ml` optimization)

### Installation

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd ias_project
    ```

2. **Create and activate a virtual environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

### Gurobi License (Important!)

The `advanced` and `advanced_ml` optimization strategies rely on the Gurobi optimization solver, which requires a license.

1. Obtain a Gurobi license (e.g., a free academic license or a commercial one).
2. Create a directory named `.secrets` in the root of the project directory.
3. Place your Gurobi license file (`gurobi.lic`) inside the `.secrets` directory:

    ```text
    ias_project/
    └── .secrets/
        └── gurobi.lic
    ```

    The simulation code (`src/light.py`) expects the license file at this location when initializing the Gurobi environment for the advanced optimizers. If the license is not found or invalid, these optimization modes will fail. The `simple` and `none` optimization modes do not require Gurobi.

## Usage

### Interactive Simulation (Streamlit App)

Run the Streamlit application for interactive simulation and visualization:

* **Using Python:**

    ```bash
    streamlit run app.py
    ```

    Open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`).

* **Using Docker Compose:**

    ```bash
    docker-compose --profile app up --build
    ```

    Open your web browser to `http://localhost:8501`.

### Script-Based Simulation

Run simulations non-interactively using scripts, primarily for data generation. Execute these from the project root directory.

* **Single Simulation Run:**
    Use `scripts/sim.py` with command-line arguments to control parameters.

    ```bash
    # Example: Run with 10 intersections, 50 cars, advanced_ml optimization for 200 steps
    python scripts/sim.py -i 10 -c 50 -o advanced_ml -s 200
    ```

    See `python scripts/sim.py --help` for all options. Data is saved in `data/<timestamp>/`.

* **Batch Simulations:**
    Use the provided bash scripts for running multiple configurations:

    ```bash
    # Run simulations with varying N intersections, borders, cars
    bash scripts/sim_configs.sh

    # Run simulations comparing different optimization types
    bash scripts/sim_opt_types.sh
    ```

    Modify the scripts directly to change the parameter ranges.

### ML Model Training

Follow the steps outlined in `ml_training/README.md`:

1. **Generate Data:** Use the simulation scripts (see above) to create sufficient data in the `data/` directory.
2. **Aggregate Data:** Run the `ml_training/aggregate_data.ipynb` notebook to process and combine the raw simulation data into `ml_training/data.parquet`.
3. **Train Model:** Run the appropriate training notebook (`model_selection.ipynb`, `neural_net.ipynb`, or `xgb_regressor.ipynb`). This will use MLflow for tracking and save the final model and scaler to the `artifacts/` directory (`model.pkl`, `scaler.pkl`).

### Running Tests

* **Using Docker Compose:**

    ```bash
    docker-compose --profile test up --build
    ```

* **Using Python:**

    ```bash
    python -m unittest discover -s tests
    ```

## Dependencies

Key libraries used in this project include:

* **Mesa:** Agent-Based Modeling framework.
* **NetworkX:** Graph creation, manipulation, and algorithms.
* **Streamlit:** Web application framework for visualization.
* **Plotly:** Interactive plotting library.
* **Polars:** High-performance DataFrame library for data collection.
* **Gurobipy:** Python interface for the Gurobi Optimizer (requires license).
* **PyOptInterface:** Interface for optimization solvers (HiGHS, Gurobi).
* **scikit-learn, XGBoost, TensorFlow/Keras:** Machine learning libraries for model training and prediction.
* **Joblib:** For saving/loading ML model artifacts.

See `requirements.txt` for the full list of dependencies.

## Configuration

* **Simulation Parameters:** Can be adjusted via the "Settings" popover in the Streamlit app (`app.py`) or through command-line arguments in `scripts/sim.py`.
* **ML Model:** The `Regressor` class (`src/regressor.py`) loads the model and scaler from the `artifacts/` directory by default.
* **Gurobi License:** Must be placed in `.secrets/gurobi.lic`.
