import logging as log
import numpy as np
import time

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from beamngpy.api.beamng import ControlApi
from beamngpy.tools import Terrain_Importer

from self_driving.decal_road import DecalRoad
from self_driving.road_points import List4DTuple, RoadPoints
from self_driving.simulation_data import SimulationParams
from self_driving.beamng_pose import BeamNGPose


class BeamNGCamera:
    def __init__(self, beamng: BeamNGpy, name: str, camera: Camera = None):
        self.name = name
        self.pose: BeamNGPose = BeamNGPose()
        self.camera = camera
        if not self.camera:
            self.camera = Camera(
                str='cam1',
                bng=beamng,
                pos=(0, 0, 0),
                dir=(0, 0, 0),
                field_of_view_y=120,
                resolution=(1280, 1280),
                is_render_colours=True,
                is_render_depth=True,
                is_render_annotations=True
                )
        self.beamng = beamng

    def get_rgb_image(self):
        self.camera.pos = self.pose.pos
        self.camera.direction = self.pose.rot
        cam_data = self.camera.poll()
        return cam_data['colour'].convert('RGB')


class BeamNGBrewer:
    def __init__(self, beamng_home=None, beamng_user=None, road_nodes: List4DTuple = None):
        self.scenario = None

        # This is used to bring up each simulation without restarting the simulator
        self.beamng: BeamNGpy = BeamNGpy('localhost', 64256, home=beamng_home, user=beamng_user)
        self.beamng.open(launch=True)

        # We need to wait until this point otherwise the BeamNG logger level will be (re)configured by BeamNGpy
        log.info("Disabling BEAMNG logs")
        for id in ["beamngpy", "beamngpy.beamngpycommon", "beamngpy.BeamNGpy", "beamngpy.beamng", "beamngpy.Scenario",
                   "beamngpy.Vehicle", "beamngpy.Camera"]:
            logger = log.getLogger(id)
            logger.setLevel(log.CRITICAL)
            logger.disabled = True

        self.vehicle: Vehicle = None
        if road_nodes:
            self.setup_road_nodes(road_nodes)

        steps = 80
        self.params = SimulationParams(beamng_steps=steps, delay_msec=int(steps * 0.05 * 1000))
        self.vehicle_start_pose = BeamNGPose()

    def setup_road_nodes(self, road_nodes):
        self.road_nodes = road_nodes
        self.decal_road: DecalRoad = DecalRoad('street_1').add_4d_points(road_nodes)
        self.road_points = RoadPoints().add_middle_nodes(road_nodes)

    def setup_vehicle(self) -> Vehicle:
        assert self.vehicle is None
        self.vehicle = Vehicle('ego_vehicle', model='etk800', licence='TIG', color='Red')
        return self.vehicle
    
    def modify_terrain(self, road_points, control: ControlApi, fade_dist=15):
        terrain_z = -28.0
        road3d = np.array([[x, y, z] for x,y,z,_ in road_points])
        road2d = road3d[:, :2]
        road_z  = road3d[:, 2]

        total_width = road_points[0][3] + 10
        half_width = total_width/2

        def height_at(point):
            dist = np.hypot(road2d[:,0] - point[0], road2d[:,1] - point[1])
            idx_min = np.argmin(dist)

            if idx_min == 0:
                idx0, idx1 = 0, 1
            elif idx_min == len(road2d) - 1:
                idx0, idx1 = idx_min - 1, idx_min
            else:
                d_prev = np.linalg.norm(road2d[idx_min - 1] - point)
                d_next = np.linalg.norm(road2d[idx_min + 1] - point)
                if d_prev < d_next:
                    idx0, idx1 = idx_min - 1, idx_min
                else:
                    idx0, idx1 = idx_min, idx_min + 1

            p0, p1 = road2d[idx0], road2d[idx1]
            z0, z1 = road_z[idx0], road_z[idx1]

            seg_vec = p1 - p0
            seg_len = np.linalg.norm(seg_vec)
            if seg_len == 0:
                t = 0
            else:
                t = np.clip(np.dot(point - p0, seg_vec) / (seg_len ** 2), 0, 1)

            z_interp = z0 + (z1 - z0) * t

            closest = p0 + t * seg_vec
            d = np.linalg.norm(point - closest)

            if d <= half_width:
                return (z_interp, True) if d <= (road_points[0][3] / 2) else (z_interp, False)
            elif d <= half_width + fade_dist:
                fade = (d - half_width) / fade_dist
                return (z_interp * (1 - fade) + terrain_z * fade, False)
            else:
                return (terrain_z, False)
        
        x_min = road2d[:, 0].min() - (half_width + fade_dist)
        x_max = road2d[:, 0].max() + (half_width + fade_dist)
        y_min = road2d[:, 1].min() - (half_width + fade_dist)
        y_max = road2d[:, 1].max() + (half_width + fade_dist)

        control.queue_lua_command('terrain = scenetree.findObject("terrain1")')
        control.queue_lua_command('road = scenetree.findObject("street_1")')
        control.queue_lua_command('road:unregisterObject()')
        lua_chunk = '(function() '
        for i in range(int(np.floor(x_min)), int(np.ceil(x_max))+1):
            for j in range(int(np.floor(y_min)), int(np.ceil(y_max))+1):
                h, _ = height_at([i, j])
                pos = f'Point3F({i}, {j}, -28.0)'
                lua_chunk += f'terrain:setHeightWs({pos}, {h}); '
        lua_chunk += 'end)()'
        control.queue_lua_command(lua_chunk)
        control.queue_lua_command('terrain:updateGrid()')
        control.queue_lua_command('terrain:updateGridMaterials()')
        control.queue_lua_command('road:regenerate()')
        control.queue_lua_command('road:registerObject("road")')
        control.queue_lua_command('road:regenerate()')

    def bring_up(self, road_points):

        # After 1.18 to make a scenario one needs a running instance of BeamNG
        self.scenario = Scenario('tig', 'tigscenario')
        if self.vehicle:
            self.scenario.add_vehicle(self.vehicle, pos=self.vehicle_start_pose.pos,
                                      rot_quat=self.vehicle_start_pose.rot, cling=True)
            
        self.scenario.make(self.beamng)
        self.beamng.set_deterministic()
        # self.beamng.set_steps_per_second(120)  # Set simulator to 60hz temporal resolution
        # self.beamng.remove_step_limit()
        self.beamng.scenario.load(self.scenario)

        self.modify_terrain(road_points, self.beamng.control)

        self.beamng.scenario.start()

        self.beamng.control.pause()
