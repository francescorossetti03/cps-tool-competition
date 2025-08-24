# This code is used in the paper
# "Model-based exploration of the frontier of behaviours for deep learning system testing"
# by V. Riccio and P. Tonella
# https://doi.org/10.1145/3368089.3409730

from random import randint, uniform
from typing import List, Tuple

from shapely.geometry import Point
from self_driving.bbox import RoadBoundingBox
from code_pipeline.tests_generation import RoadTestFactory

import math
import numpy as np
import logging as log

from self_driving.road_polygon import RoadPolygon

Tuple4F = Tuple[float, float, float, float]
Tuple2F = Tuple[float, float]

import os
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

def plot_nodes(nodes: List[Tuple[float, float, float, float]], 
               interpolated_nodes: Optional[List[Tuple[float, float, float, float]]] = None):

    x_main = [n[0] for n in nodes]
    y_main = [n[1] for n in nodes]

    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots()

    if interpolated_nodes and len(interpolated_nodes) > 0:
        x_interp = [n[0] for n in interpolated_nodes]
        y_interp = [n[1] for n in interpolated_nodes]
        ax.scatter(x_interp, y_interp, color='gray', s=10, zorder=1)

    ax.scatter(x_main, y_main, color='red', s=30, zorder=2)

    ax.set_xlim(-20, 220)
    ax.set_ylim(-20, 220)
    inner_square = plt.Rectangle((0, 0), 200, 200, fill=False, edgecolor='black', linewidth=1, zorder=0)
    ax.add_patch(inner_square)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    epoch_time = int(time.time())
    filename = f"{epoch_time}_inter.png" if interpolated_nodes and len(interpolated_nodes) > 0 else f"{epoch_time}.png"
    save_path = os.path.join("plots", filename)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Immagine salvata in: {save_path}")

def catmull_rom_spline(p0, p1, p2, p3, num_points=20):
    """p0, p1, p2, and p3 should be (x,y,z) point pairs that define the Catmull-Rom spline.
    num_points is the number of points to include in this curve segment."""
    # Convert the points to numpy so that we can do array multiplication
    p0, p1, p2, p3 = map(np.array, [p0, p1, p2, p3])

    # Calculate t0 to t4
    # For knot parametrization
    alpha = 0.5

    def tj(ti, pi, pj):
        return (((np.linalg.norm(pj - pi)) ** alpha) + ti)

    # Knot sequence
    t0 = 0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    # Interpolation points between p1 and p2
    t = np.linspace(t1, t2, num_points).reshape(num_points, 1)

    # Interpolation
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3

    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3

    c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
    return c

def catmull_rom_chain(points: List[tuple], num_spline_points=20) -> List:
    """Calculate Catmull-Rom for a chain of points and return the combined curve."""
    # The curve cr will contain an array of (x, y, z) points.
    cr = []
    for i in range(len(points) - 3):
        c = catmull_rom_spline(points[i], points[i + 1], points[i + 2], points[i + 3], num_spline_points)
        if i > 0:
            c = np.delete(c, [0], axis=0)  # Avoid duplicated point
        cr.extend(c)
    return cr

def catmull_rom(points: List[tuple], num_spline_points=20) -> List[tuple]:
    """
    Accepts points with format (x, y, z, width), returns smoothed spline
    as list of (x, y, z, width).
    """
    if len(points) < 4:
        raise ValueError("points should have at least 4 points")
    assert all(p[3] == points[0][3] for p in points), "All points must have same width"
    
    np_point_array = catmull_rom_chain([(p[0], p[1], p[2]) for p in points], num_spline_points)

    width = points[0][3]
    return [(p[0], p[1], p[2], width) for p in np_point_array]


class RoadGenerator:
    """Generate random roads given the configuration parameters"""

    NUM_INITIAL_SEGMENTS_THRESHOLD = 2
    NUM_UNDO_ATTEMPTS = 20
    MAX_Z_VARIATION = 6.0  # max variation for segment
    MAX_SLOPE = 0.3  # max slope/distance ratio
    GROUND_LEVEL = -28.0

    def __init__(self, num_control_nodes=15, max_angle=None, seg_length=None,
                 num_spline_nodes=None, initial_node=(125.0, 0.0, -28.0, 8.0),
                 bbox_size=(0, 0, 250, 250)):
        assert num_control_nodes > 1 and num_spline_nodes > 0
        assert 0 <= max_angle <= 360
        assert seg_length > 0
        assert len(initial_node) == 4 and len(bbox_size) == 4
        self.num_control_nodes = num_control_nodes
        self.num_spline_nodes = num_spline_nodes
        self.initial_node = initial_node
        self.max_angle = max_angle
        self.seg_length = seg_length
        self.road_bbox = RoadBoundingBox(bbox_size)
        assert not self.road_bbox.intersects_vertices(self._get_initial_point())

    def generate_control_nodes(self, attempts=NUM_UNDO_ATTEMPTS) -> List[Tuple4F]:
        condition = True
        while condition:
            nodes = [self._get_initial_control_node(), self.initial_node]
            i = 0
            attempt = 0

            while i < self.num_control_nodes and attempt <= attempts:
                nodes.append(self._get_next_node(nodes[-2], nodes[-1], self._get_next_max_angle(i)))
                road_polygon = RoadPolygon.from_nodes(nodes)
                budget = self.num_control_nodes - i
                assert budget >= 1

                intersect_boundary = self.road_bbox.intersects_boundary(road_polygon.polygons[-1])
                is_valid = (
                    road_polygon.is_valid() and
                    (((i == 0) and intersect_boundary) or ((i > 0) and not intersect_boundary)) and
                    self._is_valid_slope(nodes[-2], nodes[-1])
                )

                while not is_valid and budget > 0:
                    nodes.pop()
                    budget -= 1
                    attempt += 1
                    nodes.append(self._get_next_node(nodes[-2], nodes[-1], self._get_next_max_angle(i)))
                    road_polygon = RoadPolygon.from_nodes(nodes)
                    intersect_boundary = self.road_bbox.intersects_boundary(road_polygon.polygons[-1])
                    is_valid = (
                        road_polygon.is_valid() and
                        (((i == 0) and intersect_boundary) or ((i > 0) and not intersect_boundary)) and
                        self._is_valid_slope(nodes[-2], nodes[-1])
                    )

                if is_valid:
                    i += 1
                else:
                    assert budget == 0
                    nodes.pop()
                    if len(nodes) > 2:
                        nodes.pop()
                        i -= 1

                assert RoadPolygon.from_nodes(nodes).is_valid()
                assert 0 <= i <= self.num_control_nodes

            if len(nodes) - 2 == self.num_control_nodes:
                condition = False
        return nodes

    def is_valid(self, control_nodes, sample_nodes, num_spline_nodes):
        return (RoadPolygon.from_nodes(sample_nodes).is_valid() and
                self.road_bbox.contains(RoadPolygon.from_nodes(control_nodes[1:-1])))

    def generate(self):
        sample_nodes = None
        while not self.is_valid(control_nodes, sample_nodes, self.num_spline_nodes):
            control_nodes = self.generate_control_nodes()
            control_nodes = control_nodes[1:]
            control_nodes[1] = (control_nodes[1][0], control_nodes[1][1], control_nodes[0][2], control_nodes[1][3])
            sample_nodes = catmull_rom(control_nodes, self.num_spline_nodes)
        road = [(node[0], node[1], node[2]) for node in sample_nodes]
        return road

    def _get_initial_point(self) -> Point:
        return Point(self.initial_node[0], self.initial_node[1])

    def _get_initial_control_node(self) -> Tuple4F:
        x0, y0, z0, width = self.initial_node
        x, y = self._get_next_xy(x0, y0, 270)
        z = self._get_next_z(z0)
        return x, y, z, width

    def _get_next_node(self, first_node, second_node: Tuple4F, max_angle) -> Tuple4F:
        v = np.subtract(second_node[:2], first_node[:2])
        start_angle = int(np.degrees(np.arctan2(v[1], v[0])))
        angle = randint(start_angle - max_angle, start_angle + max_angle)
        x0, y0, z0, width0 = second_node
        x1, y1 = self._get_next_xy(x0, y0, angle)
        z1 = self._get_next_z(z0)
        return x1, y1, z1, width0

    def _get_next_xy(self, x0: float, y0: float, angle: float) -> Tuple2F:
        angle_rad = math.radians(angle)
        return x0 + self.seg_length * math.cos(angle_rad), y0 + self.seg_length * math.sin(angle_rad)

    def _get_next_z(self, z0: float) -> float:
        return max((z0 + uniform(-self.MAX_Z_VARIATION, self.MAX_Z_VARIATION)), self.GROUND_LEVEL)

    def _get_next_max_angle(self, i: int, threshold=NUM_INITIAL_SEGMENTS_THRESHOLD) -> float:
        if i < threshold or i == self.num_control_nodes - 1:
            return 0
        else:
            return self.max_angle

    def _is_valid_slope(self, prev_node, new_node) -> bool:
        dz = abs(new_node[2] - prev_node[2])
        dx = math.hypot(new_node[0] - prev_node[0], new_node[1] - prev_node[1])
        if dx == 0:
            return False
        return (dz / dx) <= self.MAX_SLOPE

class JanusGenerator():
    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def start(self):
        log.info("Starting test generation")
        # Generate a single test.

        #TODO: return points
        NODES = 10
        MAX_ANGLE = 40
        NUM_SPLINE_NODES = 20
        SEG_LENGTH = 25
        test_outcome = None
        count = 0

        # while(test_outcome != "FAIL"):
        while not self.executor.is_over():
            road_points = RoadGenerator(num_control_nodes=NODES, max_angle=MAX_ANGLE, seg_length=SEG_LENGTH,
                                 num_spline_nodes=NUM_SPLINE_NODES).generate()

            log.info("Generated test from road points %s", road_points)
            the_test = RoadTestFactory.create_road_test(road_points)

            test_outcome, description, execution_data = self.executor.execute_test(the_test)

            log.info("Test outcome: {}. Description: {}".format(test_outcome, description))
            count += 1
            log.info("Remaining Time: %s", str(self.executor.get_remaining_time()["time-budget"]))

            log.info("Successful tests: %s", str(count))


if __name__ == "__main__":
    # TODO Clean up the code and remove hardcoded logic from the sample generators. Create a unit tests instead
    time_budget = 250000
    map_size = 250
    beamng_home = r"C:\Users\vinni\bng_competition\BeamNG.research.v1.7.0.0"

    from code_pipeline.beamng_executor import BeamngExecutor

    executor = BeamngExecutor(time_budget=time_budget, map_size=map_size, beamng_home=beamng_home)

    roadgen = JanusGenerator(time_budget, executor, map_size)
    roadgen.start()
