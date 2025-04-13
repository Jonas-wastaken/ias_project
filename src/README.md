# Traffic Simulation Source Files

This directory contains the core Python source code for the agent-based traffic simulation model.

## Overview

The simulation models traffic flow in a grid-like urban environment. It uses the Mesa framework for agent-based modeling. Cars (`CarAgent`) navigate a graph-based road network (`Graph`), interacting with traffic lights (`LightAgent`) at intersections. Different optimization strategies can be applied to the traffic lights to manage traffic flow. Data is collected throughout the simulation using Polars DataFrames managed by various `SimData` subclasses. An optional ML regressor (`Regressor`) can be used for advanced traffic prediction. Visualization is handled by `TrafficGraph` using Plotly.

## Files

* **`model.py`**:
  * Defines the main `TrafficModel` class, which orchestrates the simulation.
  * Manages the creation, scheduling, and interaction of `CarAgent` and `LightAgent` instances.
    * Includes data collection classes (`LightIntersectionMapping`, `LightData`, `NumCars`, `Connections`, `GlobalWaitTimes`) for tracking simulation state and results.
    * Handles car respawning logic and different light optimization modes.
* **`car.py`**:
  * Defines the `CarAgent` class, representing individual vehicles.
    * Handles car movement, pathfinding (using NetworkX Dijkstra), and interaction with traffic lights.
    * Includes `WaitTimes` dataclass for tracking individual car wait times at lights.
    * Defines the `AgentArrived` exception for when a car reaches its destination.
* **`light.py`**:
  * Defines the `LightAgent` class, representing traffic lights at intersections.
  * Manages the state of the traffic light (open lane, cooldown).
    * Includes `Optimizer` abstract base class and concrete implementations (`SimpleOptimizer`, `AdvancedOptimizer`) for different light control strategies.
    * Contains helper dataclasses `Lanes` (lane distances) and `TrafficData` (historical traffic counts).
    * Defines the `LightCooldown` exception.
* **`graph.py`**:
  * Defines the `Graph` class, extending `networkx.Graph`, to represent the road network.
    * Manages 'intersection' and 'border' nodes and weighted edges (roads).
    * Provides methods for building, modifying, querying, saving, and loading the graph structure.
* **`data.py`**:
  * Defines the abstract base class `SimData`.
    * Provides a standard interface for all data collection classes used within the simulation, ensuring they use Polars DataFrames and have methods for initialization, updating, and retrieval.
* **`regressor.py`**:
  * Defines the `Regressor` class.
    * Loads a pre-trained StandardScaler and XGBoost model (`.pkl` files expected in an `artifacts` directory).
    * Provides methods to scale input features and make predictions, used by the `advanced_ml` optimization mode in `LightAgent`.
* **`graph_viz.py`**:
  * Defines the `TrafficGraph` class, inheriting from `plotly.graph_objects.Figure`.
    * Visualizes the `TrafficModel` state, including the network graph, car positions, and traffic light states (open lanes shown as arrows) using Plotly and NetworkX for layout.

## Key Dependencies

* **Mesa:** Core agent-based modeling framework.
* **NetworkX:** Graph creation, manipulation, and analysis (pathfinding, centrality).
* **Polars:** Efficient data manipulation for simulation data collection.
* **Plotly:** Interactive visualization of the simulation state.
* **NumPy:** Numerical operations.
* **scikit-learn / XGBoost:** Used by `regressor.py` for ML predictions (optional).
* **Joblib:** Used by `regressor.py` for loading model artifacts.
* **PyOptInterface:** Used by `light.py` optimizers to interface with HiGHS and Gurobi solvers.