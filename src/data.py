"""
data.py

This module defines the abstract base class `SimData` for managing and structuring
simulation data collected during a `TrafficModel` run. It uses the Polars library
for efficient data handling.

Subclasses inheriting from `SimData` are expected to implement methods for
initializing the data schema, updating the data during the simulation, and
retrieving the collected data as a Polars DataFrame.

Classes:
    - SimData: An abstract base class defining the interface for simulation data containers.

Dependencies:
    - abc: Used to define the abstract base class and methods.
    - dataclasses: Used for creating data classes (though `SimData` itself is abstract).
    - polars: Used for creating and manipulating DataFrames.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl


@dataclass
class SimData(ABC):
    """Abstract base class for simulation data containers.

    Provides a template for classes designed to store and manage data collected
    during a `TrafficModel` simulation run. Subclasses must implement the
    abstract methods defined here.
    """

    @abstractmethod
    def __post_init__(self):
        """Initializes the internal data structure (e.g., DataFrame schema).

        This method is called automatically after the dataclass is initialized
        and should be used to set up the necessary data structures, like defining
        the columns of a Polars DataFrame.
        """
        pass

    @abstractmethod
    def update_data(self) -> None:
        """Updates the stored data with new information.

        This method should be implemented by subclasses to append or modify the
        simulation data as it becomes available during the simulation steps.
        """
        pass

    @abstractmethod
    def get_data(self) -> pl.DataFrame:
        """Returns the collected simulation data.

        Returns:
            pl.DataFrame: A Polars DataFrame containing the data collected
                          by the instance.
        """
        pass
