from random import randint, uniform
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep

import logging as log
from typing import Tuple
from math import dist


class RandomTestGenerator():
    """
        This simple (naive) test generator creates roads using 4 points randomly placed on the map.
        We expect that this generator quickly creates plenty of tests, but many of them will be invalid as roads
        will likely self-intersect.
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def start(self):

        DIST_THRESHOLD_LOW = 5.0
        DIST_THRESHOLD_MED = 20.0

        DELTA_Z_LOW = 2.0
        DELTA_Z_MED = 5.0
        DELTA_Z_HIGH = 10.0

        GROUND_LEVEL = -28.0  

        def get_z(p0: Tuple[float, float, float], p1: Tuple[float, float]) -> float:
            prev_x, prev_y, prev_z = p0
            curr_x, curr_y = p1

            d = dist((prev_x, prev_y), (curr_x, curr_y))

            if d < DIST_THRESHOLD_LOW:
                delta_z = uniform(-DELTA_Z_LOW, DELTA_Z_LOW)
            elif d < DIST_THRESHOLD_MED:
                delta_z = uniform(-DELTA_Z_MED, DELTA_Z_MED)
            else:
                delta_z = uniform(-DELTA_Z_HIGH, DELTA_Z_HIGH)

            new_z = prev_z + delta_z
            return round(max(new_z, GROUND_LEVEL), 3)

        while not self.executor.is_over():
            # Some debugging
            time_remaining = self.executor.get_remaining_time()["time-budget"]
            log.info(f"Starting test generation. Remaining time {time_remaining}")

            # Simulate the time to generate a new test
            sleep(0.5)
            # Pick up random points from the map. They will be interpolated anyway to generate the road
            road_points = []
            for i in range(3):
                x = randint(0, self.map_size)
                y = randint(0, self.map_size)
                z = GROUND_LEVEL if i == 0 else get_z(road_points[-1], (x, y))
                road_points.append((x, y, z))

            log.info("Generated test using: %s", road_points)
            # Decorate the_test object with the id attribute
            the_test = RoadTestFactory.create_road_test(road_points)

            time_remaining = self.executor.get_remaining_time()["time-budget"]
            log.info(f"Simulated test generation for 0.5 sec. Remaining time {time_remaining}")
            # Try to execute the test
            test_outcome, description, execution_data = self.executor.execute_test(the_test)
            time_remaining = self.executor.get_remaining_time()["time-budget"]
            log.info(f"Executed test {the_test.id}. Remaining time {time_remaining}")

            # Print the result from the test and continue
            log.info("test_outcome %s", test_outcome)
            log.info("description %s", description)


