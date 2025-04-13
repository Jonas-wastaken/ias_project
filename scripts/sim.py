"""
sim.py

This script runs a traffic simulation using the TrafficModel class. It allows users to
configure the simulation parameters via command-line arguments, such as the number of
intersections, cars, borders, and optimization type. The simulation collects data on
traffic patterns, wait times, and other metrics, and stores the results in a structured
directory with timestamped filenames.

Classes:
    - DataPath: Manages the directory and file paths for storing simulation data.
    - Sim: Handles the initialization, execution, and data collection of the traffic simulation.

Functions:
    - parse_args: Parses command-line arguments to configure the simulation.

Usage:
    Run the script from the command line with optional arguments to customize the simulation.

    **Important**: Run from parent directory of the project.

    The simulation results, including configuration and collected data, will be stored in
    a timestamped directory under the "data" folder.
"""

import argparse
import datetime
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from src.model import TrafficModel


def parse_args() -> dict:
    """Parses command line arguments for the traffic simulation configuration.

    Returns:
        dict: A dictionary containing the parsed command line arguments,
              with default values applied where necessary.
    """
    parser = argparse.ArgumentParser(description="Traffic Simulation Configuration")
    parser.add_argument(
        "-i",
        "--num_intersections",
        type=int,
        default=random.randint(50, 125),
        help="Number of intersections",
    )
    parser.add_argument(
        "-c", "--num_cars", type=int, default=None, help="Number of cars"
    )
    parser.add_argument(
        "-b", "--num_borders", type=int, default=None, help="Number of borders"
    )
    parser.add_argument(
        "-m", "--min_distance", type=int, default=None, help="Minimum distance"
    )
    parser.add_argument(
        "-x", "--max_distance", type=int, default=None, help="Maximum distance"
    )
    parser.add_argument(
        "-o",
        "--optimization_type",
        type=str,
        default="advanced_ml",
        help="Optimization technique used for the LightAgents. One of: ['none', 'simple', 'advanced', 'advanced_ml']",
    )
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of steps")

    config = vars(parser.parse_args())

    if config["num_cars"] is None:
        config["num_cars"] = int(
            round((config["num_intersections"] * random.uniform(5, 10)), 0)
        )
    if config["num_borders"] is None:
        config["num_borders"] = int(
            round((config["num_intersections"] * random.randint(3, 5)), 0)
        )
    if config["min_distance"] is None:
        config["min_distance"] = random.randint(5, 10)
    if config["max_distance"] is None:
        config["max_distance"] = int(
            round((config["min_distance"] * random.randint(2, 4)), 0)
        )

    return config


@dataclass
class DataPath:
    """Manages the directory and file paths for storing simulation data.

    Creates a timestamped directory within the 'data' folder upon instantiation.

    Attributes:
        path (Path): The Path object representing the timestamped data directory.
    """

    path: Path = field(
        default_factory=lambda: Path.joinpath(
            Path.cwd(),
            "data",
            datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
        )
    )

    def __post_init__(self):
        """Creates the data directory if it doesn't exist."""
        self.path.mkdir(parents=True, exist_ok=True)

    def get_path(self) -> Path:
        """Returns the Path object for the data directory.

        Returns:
            Path: The path to the simulation data directory.
        """
        return self.path

    def get_file_path(self, filename: str) -> Path:
        """Constructs the full path for a file within the data directory.

        Args:
            filename (str): The name of the file (e.g., 'traffic.parquet').

        Returns:
            Path: The full path to the specified file within the data directory.
        """
        file_path = Path.joinpath(self.path, filename)
        return file_path


class Sim:
    """Handles the initialization, execution, and data saving of a traffic simulation.

    Instantiating this class runs the entire simulation process based on the
    provided configuration.

    Attributes:
        data_path (DataPath): An instance of DataPath managing the output directory.
    """

    def __init__(self, config: dict):
        """Initializes and runs the traffic simulation, then saves the results.

        Args:
            config (dict): A dictionary containing the simulation configuration
                           parameters (e.g., num_cars, steps, optimization_type).
        """
        model = TrafficModel(
            num_cars=config["num_cars"],
            optimization_type=config["optimization_type"],
            num_intersections=config["num_intersections"],
            num_borders=config["num_borders"],
            min_distance=config["min_distance"],
            max_distance=config["max_distance"],
        )

        start_time = time.time()

        for _ in tqdm(range(config["steps"]), desc="Running simulation", unit="step"):
            model.step()

        print(100 * "-")
        print("Sim completed!")
        print(
            f"Avg. time per step: {round(((time.time() - start_time) / model.steps), 2)} seconds"
        )

        self.data_path = DataPath()
        model.DataCollector(
            agents=model.get_agents_by_type("LightAgent"), data_name="traffic"
        ).get_data().write_parquet(file=self.data_path.get_file_path("traffic.parquet"))
        model.global_wait_times.get_data().write_parquet(
            file=self.data_path.get_file_path("wait_times.parquet")
        )
        model.light_intersection_mapping.get_data().write_parquet(
            file=self.data_path.get_file_path("light_intersection_mapping.parquet")
        )
        model.light_data.get_data().write_parquet(
            file=self.data_path.get_file_path("light_data.parquet")
        )
        model.n_cars.get_data().write_parquet(
            file=self.data_path.get_file_path("n_cars.parquet")
        )
        model.connections.get_data().write_parquet(
            file=self.data_path.get_file_path("connections.parquet")
        )


if __name__ == "__main__":
    config = parse_args()
    print("Starting simulation with config:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print(100 * "-")

    sim = Sim(config)

    with open(file=sim.data_path.get_file_path("config.json"), mode="w") as file:
        json.dump(obj=config, fp=file, indent=4)

    print(f"Simulation data stored in: {sim.data_path.get_path()}")
