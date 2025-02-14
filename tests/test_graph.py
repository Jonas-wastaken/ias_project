import os
import unittest
import sys
import logging

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from graph import Graph

log_dir = os.path.join("tests", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "test_graph.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class TestGraph(unittest.TestCase):
    """Unit tests for the Graph class.

    - This class contains unit tests for the Graph class, which is part of the TrafficModel.
    - It tests the addition of intersections and borders, the connections between intersections and borders, and the placement and movement of agents in the graph.

    Attributes:
        num_intersections (int): Number of intersections in the graph.
        num_borders (int): Number of borders in the graph.
        min_distance (int): Minimum distance between intersections.
        max_distance (int): Maximum distance between intersections.
        graph (Graph): Instance of the Graph.
    """
    def setUp(self):
        self.num_intersections = 10
        self.num_borders = 3
        self.min_distance = 1
        self.max_distance = 10
        self.graph = Graph(
            self.num_intersections,
            self.num_borders,
            self.min_distance,
            self.max_distance,
        )
        logging.info("Setup complete: Graph initialized")

    def test_add_intersections(self):
        """Test the addition of intersections to the graph.

        - This test checks if the number of nodes in the graph that start with "intersection" matches the expected number of intersections.
            - Collect all nodes in the graph that start with "intersection".
            - Assert that the number of collected intersections matches the expected number.

        Raises:
            AssertionError: If the number of intersections does not match the expected number.
        """
        intersections = [
            node for node in self.graph.nodes if node.startswith("intersection")
        ]
        logging.info(f"Intersections added: {intersections}")
        try:
            self.assertEqual(len(intersections), self.num_intersections)
            logging.info("Passed test_add_intersections")
        except AssertionError:
            logging.error(
                f"Failed test_add_intersections: Expected {self.num_intersections}, got {len(intersections)}"
            )
            raise

    def test_add_borders(self):
        """Test the addition of borders to the graph.

        - This test checks if the number of nodes in the graph that start with "border" matches the expected number of borders.
            - Collect all nodes in the graph that start with "border".
            - Assert that the number of collected borders matches the expected number.

        Raises:
            AssertionError: If the number of borders does not match the expected number.
        """
        borders = [node for node in self.graph.nodes if node.startswith("border")]
        logging.info(f"Borders added: {borders}")
        try:
            self.assertEqual(len(borders), self.num_borders)
            logging.info("Passed test_add_borders")
        except AssertionError:
            logging.error(
                f"Failed test_add_borders: Expected {self.num_borders}, got {len(borders)}"
            )
            raise

    def test_connect_intersections(self):
        """Test the connections between intersections in the graph.

        - This test checks if each intersection is connected to at least 2 and at most 4 other intersections.
            - Collect all nodes in the graph that start with "intersection".
            - For each intersection, count the number of neighbors that are also intersections.
            - Assert that the number of connected intersections is at least 2 and at most 4.

        Raises:
            AssertionError: If an intersection is not connected to at least 2 or at most 4 other intersections.
        """
        intersections = [
            node for node in self.graph.nodes if node.startswith("intersection")
        ]
        for intersection in intersections:
            neighbors = len(
                [
                    node
                    for node in self.graph.neighbors(intersection)
                    if node.startswith("intersection")
                ]
            )
            logging.info(
                f"Intersection {intersection} connected intersections: {neighbors}"
            )
            try:
                self.assertGreaterEqual(neighbors, 2)
                self.assertLessEqual(neighbors, 4)
            except AssertionError:
                logging.error(
                    f"Failed test_connect_intersections: Intersection {intersection} is connected to {neighbors} intersections"
                )
                raise
        logging.info("Passed test_connect_intersections")

    def test_connect_borders(self):
        """Test the connections between borders in the graph.

        - This test checks if each border is connected to exactly 1 intersection.
            - Collect all nodes in the graph that start with "border".
            - For each border, count the number of neighbors that are intersections.
            - Assert that the number of connected intersections is exactly 1.

        Raises:
            AssertionError: If a border is not connected to exactly 1 intersection.
        """
        borders = [node for node in self.graph.nodes if node.startswith("border")]
        for border in borders:
            logging.info(f"Border {border} connections: {self.graph.degree(border)}")
            try:
                self.assertEqual(self.graph.degree(border), 1)
                self.assertTrue(list(self.graph.neighbors(border))[0].startswith("intersection"))
            except AssertionError:
                logging.error(
                    f"Failed test_connect_borders: Border {border} has {self.graph.degree(border)} connections"
                )
                raise
        logging.info("Passed test_connect_borders")

    def test_place_agent(self):
        """Test the placement of an agent in the graph.

        - This test checks if an agent is placed at a border node.
            - Place an agent in the graph.
            - Assert that the agent is placed at a border node.

        Raises:
            AssertionError: If the agent is not placed at a border node.
        """
        agent_id = 1
        start_node = self.graph.place_agent(agent_id)
        logging.info(f"Agent {agent_id} placed at {start_node}")
        try:
            self.assertIn(agent_id, self.graph.agent_positions[start_node])
            self.assertTrue(start_node.startswith("border"))
            logging.info("Passed test_place_agent")
        except AssertionError:
            logging.error(
                f"Failed test_place_agent: Agent {agent_id} not placed correctly at {start_node}"
            )
            raise

    def test_move_agent(self):
        """Test the movement of an agent in the graph.

        - This test checks if an agent is moved from one node to another.
            - Place an agent in the graph.
            - Move the agent to a new node.
            - Assert that the agent is moved to the new node.

        Raises:
            AssertionError: If the agent is not moved to the new node.
        """
        agent_id = 1
        start_node = self.graph.place_agent(agent_id)
        new_position = "intersection_0"
        self.graph.move_agent(agent_id, new_position)
        logging.info(f"Agent {agent_id} moved from {start_node} to {new_position}")
        try:
            self.assertEqual(self.graph.agent_positions[agent_id], new_position)
            logging.info("Passed test_move_agent")
        except AssertionError:
            logging.error(
                f"Failed test_move_agent: Agent {agent_id} not moved correctly to {new_position}"
            )
            raise

    def test_save(self):
        """Test the saving of the graph to a file.

        - This test checks if the graph is saved to a file.
            - Save the graph to a file.
            - Assert that the file exists.
            - Remove the file.

        Raises:
            AssertionError: If the file does not exist.
        """
        filename = "test_graph.pickle"
        self.graph.save(filename)
        logging.info(f"Graph saved to {filename}")
        try:
            self.assertTrue(os.path.exists(filename))
            os.remove(filename)
            logging.info(f"Graph file {filename} removed")
            logging.info("Passed test_save")
        except AssertionError:
            logging.error(f"Failed test_save: File {filename} does not exist")
            raise


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestGraph("test_add_intersections"))
    suite.addTest(TestGraph("test_add_borders"))
    suite.addTest(TestGraph("test_connect_intersections"))
    suite.addTest(TestGraph("test_connect_borders"))
    suite.addTest(TestGraph("test_place_agent"))
    suite.addTest(TestGraph("test_move_agent"))
    suite.addTest(TestGraph("test_save"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
