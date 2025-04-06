from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class Regressor:
    def __init__(self):
        self.scaler: StandardScaler = joblib.load(
            Path.joinpath(Path.cwd(), "artifacts", "scaler.pkl")
        )
        self.model: RandomForestRegressor = joblib.load(
            Path.joinpath(Path.cwd(), "artifacts", "model.pkl")
        )

    def scale_input(
        self, time: int, centrality: float, distance: int, incoming_cars: int
    ) -> np.array:
        x = np.array([time, centrality, distance, incoming_cars]).reshape(1, -1)
        x = self.scaler.transform(x)

        return x

    def predict(
        self, time: int, centrality: float, distance: int, incoming_cars: int
    ) -> int:
        y = self.model.predict(
            self.scale_input(time, centrality, distance, incoming_cars)
        )

        y = np.round(y, 0).astype(int)

        return int(y[0])
