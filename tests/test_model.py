import os
import unittest
import sys
import logging

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from model import TrafficModel
from car import CarAgent
from graph import Graph

log_dir = os.path.join("tests", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "test_model.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class TestTrafficModel(unittest.TestCase):
    def setUp(self):
        self.num_agents = 5
        self.num_intersections = 10
        self.num_borders = 3
        self.min_distance = 1
        self.max_distance = 10
        self.model = TrafficModel(
            num_agents=self.num_agents,
            num_intersections=self.num_intersections,
            num_borders=self.num_borders,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )
        logging.info("Setup complete: TrafficModel initialized")

    def test_initial_agents(self):
        """Test initial agents.

        - This test checks if the agents are initialized correctly.
            - Check if the agents are instances of the CarAgent class.
            - Check if the agent paths are initialized as dictionaries.
            - Assert that the number of agents matches the expected number.

        Raises:
            AssertionError: If the agents are not instances of the CarAgent class, the agent paths are not initialized as dictionaries, or the number of agents does not match the expected number.
        """
        logging.info("Test initial agents")
        logging.info(f"Initialized {self.num_agents} Agents!")
        try:
            for agent in self.model.agents:
                logging.info(f"Agent {agent.unique_id}")
                logging.info(f"{self.model.agent_paths[agent.unique_id]}")
                self.assertIsInstance(agent, CarAgent)
                self.assertIsInstance(self.model.agent_paths[agent.unique_id], dict)
            self.assertEqual(len(self.model.agents), self.num_agents)
            self.assertEqual(len(self.model.agent_paths), self.num_agents)
            logging.info("Passed test_initial_agents")
        except AssertionError as e:
            logging.error(f"Failed test_initial_agents: {e}")
            raise

    def test_graph_initialization(self):
        """Test graph initialization.

        - This test checks if the graph is initialized correctly.
            - Check if the grid is an instance of the Graph class.
            - Assert that the number of nodes in the graph matches the expected number of intersections and borders.

        Raises:
            AssertionError: If the grid is not an instance of the Graph class or the number of nodes in the graph does not match the expected number of intersections and borders.
        """
        logging.info("Test graph initialization")
        logging.info(f"Graph nodes: {self.model.grid.nodes}")
        try:
            self.assertIsInstance(self.model.grid, Graph)
            self.assertEqual(
                len(self.model.grid.nodes),
                self.num_intersections + self.num_borders,
            )
            logging.info("Passed test_graph_initialization")
        except AssertionError as e:
            logging.error(f"Failed test_graph_initialization: {e}")
            raise

    def test_step(self):
        """Test step.

        - This test checks if the agents move to their next positions after a step.
            - Save the initial positions of the agents.
            - Call the step method of the model.
            - Assert that the initial positions are not the same as the current positions.

        Raises:
            AssertionError: If the initial positions are the same as the current positions.
        """
        logging.info("Test step")
        initial_positions = self.model.agent_paths.copy()
        logging.info(f"{[path for path in self.model.agent_paths.values()]}")
        try:
            self.model.step()
            self.assertIsNot(initial_positions, self.model.agent_paths)
            logging.info("Passed test_step")
        except AssertionError as e:
            logging.error(f"Failed test_step: {e}")
            raise

    def test_agent_removal(self):
        """Test agent removal.

        - This test checks if the agents are removed from the model after reaching their goals.
            - Save the highest path length of the agents.
            - Call the step method of the model until there are no agents left or the steps exceed the highest path length.

        Raises:
            AssertionError: If there are agents left after the step method is called or if the steps exceed the highest path length.
        """
        logging.info("Test agent removal")
        highest_path_length = 0
        for agent in self.model.agents:
            path_length = sum(value for value in agent.path.values() if value)
            if path_length > highest_path_length:
                highest_path_length = path_length
        logging.info(f"Highest path length: {highest_path_length}")
        steps = 0
        try:
            while self.model.agents and steps <= highest_path_length:
                steps += 1
                self.model.step()
                logging.info(f"Agents left: {len(self.model.agents)}")
                for agent in self.model.agents:
                    logging.info(f"Agent {agent.unique_id}")
                    logging.info(f"Position: {agent.position}")
                    logging.info(f"Goal: {agent.goal}")
            self.assertEqual(len(self.model.agents), 0)
            logging.info("Passed test_agent_removal")
        except AssertionError as e:
            logging.error(
                f"Failed test_agent_removal. Max steps: {highest_path_length} is exceeded: {e}"
            )
            raise


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTrafficModel("test_initial_agents"))
    suite.addTest(TestTrafficModel("test_graph_initialization"))
    suite.addTest(TestTrafficModel("test_step"))
    suite.addTest(TestTrafficModel("test_agent_removal"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
