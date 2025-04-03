import polars as pl
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SimData(ABC):
    """This abstract class provides the template for all data structures related to storing data of a TrafficModel simulation run"""

    @abstractmethod
    def __post_init__(self):
        """Constructs the data schema"""
        pass

    @abstractmethod
    def update_data(self) -> None:
        """Updates the data"""
        pass

    @abstractmethod
    def get_data(self) -> pl.DataFrame:
        """Returns the data

        Returns:
            pl.DataFrame: Data
        """
        pass
