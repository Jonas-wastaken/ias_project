# ML Model Training

The machine learning model predicts the number of cars expected at a specific lane of a traffic light based on the current time, the centrality of the intersection, whether it's an entry point, and the distance to the next intersection.

## Steps

1. **Generate Simulation Data**: Run traffic simulations using the simulation script (`scripts/sim.py`) or specific scripts.
2. **Aggregate Data**: Execute the [ml_training/aggregate_data.ipynb](ml_training/aggregate_data.ipynb) notebook. This script reads simulation data from all subdirectories within `data/`, aggregates it, performs necessary joins, and saves the final dataset as `ml_training/data.parquet` and metadata as `ml_training/metadata.parquet` and [ml_training/metadata.json](ml_training/metadata.json).
3. **Explore Data (Optional)**: Run the [ml_training/explore_data.ipynb](ml_training/explore_data.ipynb) notebook to understand the characteristics and distributions within the aggregated dataset (`ml_training/data.parquet`).
4. **Model Selection & Training**:
    * The [ml_training/model_selection.ipynb](ml_training/model_selection.ipynb) notebook trains and evaluates several baseline regression models (Linear Regression, Decision Tree, Random Forest, XGBoost, KNN, Bayesian Ridge, MLP) using MLflow for tracking.
    * The [ml_training/neural_net.ipynb](ml_training/neural_net.ipynb) notebook specifically trains and tunes a TensorFlow/Keras Dense Neural Network.
    * The [ml_training/xgb_regressor.ipynb](ml_training/xgb_regressor.ipynb) notebook trains and tunes an XGBoost Regressor.

## Output Artifacts

* **`artifacts/model.pkl`**: The serialized trained machine learning model.
* **`artifacts/scaler.pkl`**: The serialized StandardScaler used to preprocess the features during training.
* **`ml_training/data.parquet`**: The aggregated dataset used for training.
* **`ml_training/metadata.parquet`**: Metadata about the simulation runs used.
* **`ml_training/metadata.json`**: Aggregated metadata summary.
* **`ml_training/mlruns/`**: MLflow tracking data, containing parameters, metrics, and artifacts for each training run.

To use the trained model, instantiate the [`Regressor`](src/regressor.py) class from [`src/regressor.py`](src/regressor.py), which automatically loads the model and scaler from the `artifacts/` directory.
