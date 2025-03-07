import argparse
from model import TrafficModel


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the traffic simulation configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Traffic Simulation Configuration")
    parser.add_argument("-c", "--num_cars", type=int, default=50, help="Number of cars")
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
        for _ in range(config.steps):
            model.step()
            if model.steps % 100 == 0:
                print(f"Completed {model.steps} of {config.steps} steps...")
        self.data_path = model.save_sim_data()


if __name__ == "__main__":
    config = parse_args()
    print(f"Starting simulation with config:\n{config}")
    sim = Sim(config)
    print(f"Simulation data stored in:\n{sim.data_path}")
    print(f"Config: {config}")
