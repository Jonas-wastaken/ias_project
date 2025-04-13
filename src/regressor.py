"""
regressor.py

This module provides the Regressor class, which loads a pre-trained StandardScaler
and a XGBRegressor model from disk using joblib. It offers methods to
scale input features and make predictions using the loaded model.

The primary use case is to predict an outcome based on input features like time,
centrality, entrypoint status, and distance, after scaling these features appropriately.

Classes:
    - Regressor: Loads scaler and model artifacts and provides methods for scaling
                 input and making predictions.

Dependencies:
    - pathlib: For handling file paths.
    - joblib: For loading pickled model and scaler objects.
    - numpy: For numerical operations, especially array handling.
    - sklearn.preprocessing.StandardScaler: For scaling input features.
    - xgboost.XGBRegressor: The regression model used.

Usage:
    Instantiate the Regressor class to load the artifacts. Then, use the `predict`
    method with the required input features to get a rounded integer prediction.

    ```python
    from regressor import Regressor

    reg = Regressor()
    prediction = reg.predict(time=100, centrality=0.5, is_entrypoint=True, distance=50)
    print(f"Predicted value: {prediction}")
    ```

Note:
    This module expects 'scaler.pkl' and 'model.pkl' to exist in the 'artifacts'
    directory relative to the current working directory where the script using
    this module is executed.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class Regressor:
    """
    Loads a pre-trained StandardScaler and XGBRegressor model for prediction.

    This class handles loading the necessary artifacts (scaler and model) from
    disk and provides methods to scale input data and make predictions using
    the loaded XGBoost model.

    Attributes:
        scaler (StandardScaler): The loaded StandardScaler object.
        model (XGBRegressor): The loaded XGBRegressor model object.
    """

    def __init__(self):
        """
        Initializes the Regressor by loading the scaler and model from disk.

        Expects 'scaler.pkl' and 'model.pkl' to be present in the 'artifacts'
        directory relative to the current working directory.
        """
        self.scaler: StandardScaler = joblib.load(
            Path.joinpath(Path.cwd(), "artifacts", "scaler.pkl")
        )
        self.model: XGBRegressor = joblib.load(
            Path.joinpath(Path.cwd(), "artifacts", "model.pkl")
        )

    def scale_input(
        self, time: int, centrality: float, is_entrypoint: bool, distance: int
    ) -> np.ndarray:
        """
        Scales the input features using the pre-loaded StandardScaler.

        Args:
            time (int): The time feature.
            centrality (float): The centrality feature.
            is_entrypoint (bool): The entrypoint status feature.
            distance (int): The distance feature.

        Returns:
            np.ndarray: A NumPy array containing the scaled features, reshaped
                        to be suitable for model prediction (1 row, 4 columns).
        """
        x = np.array([time, centrality, is_entrypoint, distance]).reshape(1, -1)
        x = self.scaler.transform(x)

        return x

    def predict(
        self, time: int, centrality: float, is_entrypoint: bool, distance: int
    ) -> int:
        """
        Makes a prediction using the loaded XGBRegressor model after scaling the input.

        Args:
            time (int): The time feature.
            centrality (float): The centrality feature.
            is_entrypoint (bool): The entrypoint status feature.
            distance (int): The distance feature.

        Returns:
            int: The rounded integer prediction from the model.
        """
        scaled_input = self.scale_input(time, centrality, is_entrypoint, distance)
        y = self.model.predict(scaled_input)

        y_rounded = np.round(y, 0).astype(int)

        return int(y_rounded[0])
