import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
import datetime
from model import TrafficModel


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the traffic simulation configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Traffic Simulation Configuration")
    parser.add_argument(
        "-c", "--num_cars", type=int, default=100, help="Number of cars"
    )
    parser.add_argument(
        "-i",
        "--num_intersections",
        type=int,
        default=100,
        help="Number of intersections",
    )
    parser.add_argument(
        "-b", "--num_borders", type=int, default=30, help="Number of borders"
    )
    parser.add_argument(
        "-min", "--min_distance", type=int, default=10, help="Minimum distance"
    )
    parser.add_argument(
        "-max", "--max_distance", type=int, default=20, help="Maximum distance"
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=10000, help="Number of steps"
    )
    return parser.parse_args()


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
            num_cars=config.num_cars,
            num_intersections=config.num_intersections,
            num_borders=config.num_borders,
            min_distance=config.min_distance,
            max_distance=config.max_distance,
        )

        start_time = time.time()

        for _ in range(config.steps):
            model.step()
            if model.steps % 10 == 0:
                print(f"Completed {model.steps} of {config.steps} steps...")
                print(
                    f"Estimated time remaining: {int(((time.time() - start_time) / model.steps) * (config.steps - model.steps))} seconds..."
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
    print(f"Starting simulation with config:\n{config}")
    print(100 * "-")
    sim = Sim(config)
    print(f"Simulation data stored in: data/{sim.data_path.get_path()}")
    print(f"Config: {config}")
