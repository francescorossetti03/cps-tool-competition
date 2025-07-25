import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon
from shapely.affinity import translate, rotate
from math import atan2, degrees

class RoadTestVisualizer:
    """
        Visualize and Plot RoadTests
    """

    little_triangle = Polygon([(10, 0), (0, -5), (0, 5), (10, 0)])
    square = Polygon([(5, 5), (5, -5), (-5, -5), (-5, 5), (5,5)])

    def __init__(self, map_size):
        self.map_size = map_size
        self.last_submitted_test_figure = None

        plt.ion()
        plt.show()

    def _setup_figure(self):
        if self.last_submitted_test_figure is not None:
            plt.figure(self.last_submitted_test_figure.number)
            plt.clf()
        else:
            self.last_submitted_test_figure = plt.figure()

        plt.gca().set_aspect('equal', 'box')
        plt.gca().set(xlim=(-30, self.map_size + 30), ylim=(-30, self.map_size + 30))

    def _add_polygon_patch(self, polygon, facecolor='gray', edgecolor='dimgray'):
        """Utility function to add a Shapely Polygon to the matplotlib plot."""
        if not polygon.is_empty:
            if polygon.geom_type == 'Polygon':
                patch = patches.Polygon(list(polygon.exterior.coords), closed=True, facecolor=facecolor, edgecolor=edgecolor)
                plt.gca().add_patch(patch)
                for interior in polygon.interiors:
                    hole_patch = patches.Polygon(list(interior.coords), closed=True, facecolor='white', edgecolor=edgecolor)
                    plt.gca().add_patch(hole_patch)
            elif polygon.geom_type == 'MultiPolygon':
                for poly in polygon.geoms:
                    self._add_polygon_patch(poly, facecolor=facecolor, edgecolor=edgecolor)

    def visualize_road_test(self, the_test):
        self._setup_figure()

        # Title: test validity
        title_string = ""
        if the_test.is_valid is not None:
            title_string = "Test is " + ("valid" if the_test.is_valid else "invalid")
            if not the_test.is_valid:
                title_string += ": " + the_test.validation_message
        plt.suptitle(title_string, fontsize=14)

        # Map boundary
        map_patch = patches.Rectangle((0, 0), self.map_size, self.map_size, linewidth=1, edgecolor='black', facecolor='none')
        plt.gca().add_patch(map_patch)

        # Road geometry (buffer around path)
        road_line = LineString([(t[0], t[1]) for t in the_test.interpolated_points])
        road_poly = road_line.buffer(8.0, cap_style=2, join_style=2)
        self._add_polygon_patch(road_poly, facecolor='gray', edgecolor='dimgray')

        # Interpolated Points
        sx = [t[0] for t in the_test.interpolated_points]
        sy = [t[1] for t in the_test.interpolated_points]
        plt.plot(sx, sy, color='yellow')

        # Road Points
        x = [t[0] for t in the_test.road_points]
        y = [t[1] for t in the_test.road_points]
        plt.plot(x, y, 'wo')

        # Starting triangle (ego start)
        delta_x = sx[1] - sx[0]
        delta_y = sy[1] - sy[0]
        angle_start = degrees(atan2(delta_y, delta_x))
        start_shape = translate(rotate(self.little_triangle, angle=angle_start, origin=(0, 0)), xoff=sx[0], yoff=sy[0])
        self._add_polygon_patch(start_shape, facecolor='black', edgecolor='black')

        # Ending square (ego end)
        delta_x = sx[-1] - sx[-2]
        delta_y = sy[-1] - sy[-2]
        angle_end = degrees(atan2(delta_y, delta_x))
        end_shape = translate(rotate(self.square, angle=angle_end, origin=(0, 0)), xoff=sx[-1], yoff=sy[-1])
        self._add_polygon_patch(end_shape, facecolor='black', edgecolor='black')

        plt.draw()
        plt.pause(0.001)
