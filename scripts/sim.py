from pathlib import Path
import sys
import argparse
import time
import random
import json
from pathlib import Path
from dataclasses import dataclass, field
import datetime

sys.path.append(Path.joinpath(Path.cwd(), Path("src")))

from src.model import TrafficModel


def parse_args() -> dict:
    """Parses command line arguments for the traffic simulation configuration.

    Returns:
        dict: Parsed command line arguments as a dictionary.
    """
    parser = argparse.ArgumentParser(description="Traffic Simulation Configuration")
    parser.add_argument(
        "-i",
        "--num_intersections",
        type=int,
        default=random.randint(25, 75),
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
            round((config["num_intersections"] * random.uniform(10, 20)), 0)
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
    """Holds a Path object representing the directory where sim data is stored in"""

    path: Path = field(
        default_factory=lambda: Path.joinpath(
            Path.cwd(),
            "data",
            datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
        )
    )

    def __post_init__(self):
        """Create the directory"""
        self.path.mkdir(parents=True, exist_ok=True)

    def get_path(self) -> Path:
        """Get directory where data is stored

        Returns:
            Path: Path object
        """
        return self.path

    def get_file_path(self, filename: str) -> Path:
        """Constructs a file path from a directory

        Args:
            filename (str): Name of the file to store

        Returns:
            Path: Path object
        """
        file_path = Path.joinpath(self.path, filename)
        return file_path


class Sim:
    """Simulation class for running the traffic model.

    Attributes:
        data_path (str): Path where the simulation data is stored.
    """

    def __init__(self, config: argparse.Namespace):
        """Initializes the simulation with the given configuration.

        Args:
            config (argparse.Namespace): Configuration parameters for the simulation.
        """
        model = TrafficModel(
            num_cars=config["num_cars"],
            optimization_type=config["optimization_type"],
            num_intersections=config["num_intersections"],
            num_borders=config["num_borders"],
            min_distance=config["min_distance"],
            max_distance=config["max_distance"],
        )

        log_interval = 10 if config["optimization_type"] in ["none", "simple"] else 1

        start_time = time.time()

        for _ in range(config["steps"]):
            model.step()
            if model.steps % log_interval == 0:
                print(f"Completed {model.steps} of {config['steps']} steps...")
                print(
                    f"Estimated time remaining: {int(((time.time() - start_time) / model.steps) * (config['steps'] - model.steps))} seconds..."
                )
                print(100 * "-")
        print("Sim completed!")
        print(
            f"Avg. time per 10 steps: {round((((time.time() - start_time) / model.steps) * 10), 2)}"
        )

        self.data_path = DataPath()
        model.DataCollector(
            agents=model.get_agents_by_type("LightAgent"), data_name="arrivals"
        ).get_data().write_parquet(
            file=self.data_path.get_file_path("arrivals.parquet")
        )
        model.DataCollector(
            agents=model.get_agents_by_type("LightAgent"), data_name="traffic"
        ).get_data().write_parquet(file=self.data_path.get_file_path("traffic.parquet"))
        model.DataCollector(
            agents=model.get_agents_by_type("CarAgent"), data_name="wait_times"
        ).get_data().write_parquet(
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
    print(f"Starting simulation with config: {config}")
    print(100 * "-")
    sim = Sim(config)
    with open(file=sim.data_path.get_file_path("config.json"), mode="w") as file:
        json.dump(obj=config, fp=file, indent=4)
    print(f"Simulation data stored in: {sim.data_path.get_path()}")
    print(f"Config: {config}")
