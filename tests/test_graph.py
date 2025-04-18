import os
import unittest
import sys
from log_config import setup_logging
import re
import random
from math import sqrt

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from graph import Graph

logger = setup_logging("test_graph")


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
        self.num_intersections = random.randint(10, 100)
        self.num_borders = random.randint(3, int(sqrt(self.num_intersections)))
        self.min_distance = 2
        self.max_distance = 10
        self.graph = Graph(
            self.num_intersections,
            self.num_borders,
            self.min_distance,
            self.max_distance,
        )

        logger.info(
            f"Test setup:\n Number of Intersections: {self.num_intersections}\n Number of Borders: {self.num_borders}\n Minimum Distance: {self.min_distance}\n Maximum Distance: {self.max_distance}"
        )

    def _test_intersections(self, expected_num_intersections: int):
        """Ensure that graph has the specified number of intersections

        Args:
            expected_num_intersections (int): Expected number of intersections

        Raises:
            AssertionError: If the graph does not have the specified number of intersections
        """
        try:
            self.assertEqual(
                len(self.graph.get_nodes("intersection")), expected_num_intersections
            )
            logger.info(f"Graph has {expected_num_intersections} intersections")
        except AssertionError as e:
            logger.error(
                f"Graph does not have {expected_num_intersections} intersections: {e}"
            )
            raise

    def _test_intersection_connections(self):
        """Ensure that intersections are connected to minimum 2 and maximum 4 other intersections

        Raises:
            AssertionError: If intersections are not connected to between 2 and 4 other intersections
        """
        try:
            intersection_pattern = re.compile(r"intersection_\d+")
            for intersection, connections in self.graph.get_connections(
                filter_by="intersection"
            ).items():
                matches = [
                    re.findall(intersection_pattern, connection)
                    for connection in connections
                ]
                flattened_matches = [item for sublist in matches for item in sublist]
                self.assertGreaterEqual(len(flattened_matches), 2)
                self.assertLessEqual(len(flattened_matches), 4)
            logger.info("Graph intersections connected correctly")
        except AssertionError as e:
            logger.error(
                f"Intersections do not have between 2 and 4 connections to other intersection nodes: {e}"
            )
            logger.info(intersection)
            logger.info(matches)
            logger.info(flattened_matches)
            raise

    def _test_borders(self, expected_num_borders: int):
        """Ensure that graph has the specified number of borders

        Args:
            expected_num_borders (int): Expected number of borders

        Raises:
            AssertionError: If the graph does not have the specified number of borders
        """
        try:
            self.assertEqual(len(self.graph.get_nodes("border")), expected_num_borders)
            logger.info(f"Graph has {expected_num_borders} borders")
        except AssertionError as e:
            logger.error(f"Graph does not have {expected_num_borders} borders: {e}")
            raise

    def _test_border_connections(self):
        """Ensure that borders are connected to exactly 1 intersection

        Raises:
            AssertionError: If borders are not connected to exactly 1 intersection
        """
        try:
            for border, connections in self.graph.get_connections(
                filter_by="border"
            ).items():
                self.assertTrue(connections[0].startswith("intersection"))
                self.assertTrue(connections[1].startswith("intersection"))
                self.assertEqual(len(connections), 2)
            logger.info("Graph borders connected correctly")
        except AssertionError as e:
            logger.error(
                f"Borders are not connected correctly: {e}\n{border}, {connections[0]}, {connections[1], {(len(connections))}}"
            )
            raise

    def test_graph_init(self):
        logger.info("Test initializing Graph")
        try:
            self.assertIsInstance(self.graph, Graph)
            self._test_intersections(expected_num_intersections=self.num_intersections)
            self._test_intersection_connections()
            self._test_borders(expected_num_borders=self.num_borders)
            self._test_border_connections()
            logger.info("Setup complete: Graph initialized")
        except AssertionError as e:
            logger.error(f"Failed to initialize Graph: {e}")
            raise

    def test_add_intersections(self):
        """Test the addition of intersections to the graph.

        - This test checks if intersections are added to the graph.
            - Add intersections to the graph.
            - Assert that the number of intersections in the graph is equal to the expected number.

        Raises:
            AssertionError: If the number of intersections in the graph is not equal to the expected number.
        """
        logger.info("Testing adding intersections to Graph")
        num_initial_intersections = len(self.graph.get_nodes("intersection"))
        num_added_intersections = 5
        self.graph.add_intersections(num_added_intersections)
        try:
            self._test_intersections(
                expected_num_intersections=num_initial_intersections
                + num_added_intersections
            )
            self._test_intersection_connections()
            logger.info("Passed test_add_intersections")
        except AssertionError as e:
            logger.error(f"Failed test_add_intersections: {e}")
            raise

    def test_add_borders(self):
        """Test the addition of borders to the graph.

        - This test checks if borders are added to the graph.
            - Add borders to the graph.
            - Assert that the number of borders in the graph is equal to the expected number.

        Raises:
            AssertionError: If the number of borders in the graph is not equal to the expected number.
        """
        logger.info("Testing adding borders to Graph")
        num_initial_borders = len(self.graph.get_nodes("border"))
        num_added_borders = 2
        self.graph.add_borders(num_added_borders)
        try:
            self._test_borders(
                expected_num_borders=num_initial_borders + num_added_borders
            )
            self._test_border_connections()
            logger.info("Passed test_add_borders")
        except AssertionError as e:
            logger.error(f"Failed test_add_borders: {e}")
            raise

    def test_place_agent(self):
        """Test the placement of an agent in the graph.

        - This test checks if an agent is placed at a border node.
            - Place an agent in the graph.
            - Assert that the agent is placed at a border node.

        Raises:
            AssertionError: If the agent is not placed at a border node.
        """
        logger.info("Testing placing agent in Graph")
        start_node = self.graph.place_agent(agent_id=1)
        try:
            self.assertTrue(start_node.startswith("border"))
            logger.info("Passed test_place_agent")
        except AssertionError as e:
            logger.error(
                f"Failed test_place_agent: Agent not placed correctly at {start_node}: {e}"
            )
            raise


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestGraph("test_graph_init"))
    suite.addTest(TestGraph("test_add_intersections"))
    suite.addTest(TestGraph("test_add_borders"))
    suite.addTest(TestGraph("test_place_agent"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
