import dataclasses
import logging
import os

import numpy as np
from idyntree.visualize import MeshcatVisualizer

import hippopt.deps.surf2stl as surf2stl
from hippopt.robot_planning.utilities.terrain_descriptor import TerrainDescriptor


@dataclasses.dataclass
class TerrainVisualizerSettings:
    terrain: TerrainDescriptor = dataclasses.field(default=None)
    terrain_color: list[float] = dataclasses.field(default=None)
    working_folder: str = dataclasses.field(default=None)
    terrain_mesh_axis_points: int = dataclasses.field(default=None)
    terrain_x_limits: list[float] = dataclasses.field(default=None)
    terrain_y_limits: list[float] = dataclasses.field(default=None)
    overwrite_terrain_files: bool = dataclasses.field(default=None)
    draw_terrain_normals: bool = dataclasses.field(default=None)
    terrain_normals_color: list[float] = dataclasses.field(default=None)
    terrain_normals_radius: float = dataclasses.field(default=None)
    terrain_normal_axis_points: int = dataclasses.field(default=None)
    terrain_normal_scaling: float = dataclasses.field(default=None)

    def __post_init__(self):
        if self.terrain_color is None:
            self.terrain_color = [0.5, 0.5, 0.5, 0.75]

        if self.working_folder is None:
            self.working_folder = "./"

        if self.terrain_mesh_axis_points is None:
            self.terrain_mesh_axis_points = 200

        if self.terrain_x_limits is None:
            self.terrain_x_limits = [-1.5, 1.5]

        if self.terrain_y_limits is None:
            self.terrain_y_limits = [-1.5, 1.5]

        if self.overwrite_terrain_files is None:
            self.overwrite_terrain_files = False

        if self.draw_terrain_normals is None:
            self.draw_terrain_normals = False

        if self.terrain_normals_color is None:
            self.terrain_normals_color = [1.0, 0.0, 0.0, 1.0]

        if self.terrain_normals_radius is None:
            self.terrain_normals_radius = 0.01

        if self.terrain_normal_axis_points is None:
            self.terrain_normal_axis_points = 20

        if self.terrain_normal_scaling is None:
            self.terrain_normal_scaling = 0.1

    def is_valid(self) -> bool:
        ok = True
        logger = logging.getLogger("[hippopt::TerrainVisualizerSettings]")
        if not os.access(self.working_folder, os.W_OK):
            logger.error("working_folder is not writable.")
            ok = False
        if len(self.terrain_color) != 4:
            logger.error("terrain_color is not specified correctly.")
            ok = False
        if self.terrain_mesh_axis_points <= 0:
            logger.error("terrain_mesh_axis_points is not specified correctly.")
            ok = False
        if self.terrain is None:
            logger.error("terrain is not specified.")
            ok = False
        if len(self.terrain_x_limits) != 2 or (
            self.terrain_x_limits[0] >= self.terrain_x_limits[1]
        ):
            logger.error("terrain_x_limits are not specified correctly.")
            ok = False
        if len(self.terrain_y_limits) != 2 or (
            self.terrain_y_limits[0] >= self.terrain_y_limits[1]
        ):
            logger.error("terrain_y_limits are not specified correctly.")
            ok = False
        if len(self.terrain_normals_color) != 4:
            logger.error("terrain_normals_color is not specified correctly.")
            ok = False
        if self.terrain_normals_radius <= 0:
            logger.error("terrain_normals_radius is not specified correctly.")
            ok = False
        if self.terrain_normal_axis_points <= 0:
            logger.error("terrain_normal_axis_points is not specified correctly.")
            ok = False
        return ok


class TerrainVisualizer:
    def __init__(
        self, settings: TerrainVisualizerSettings, viz: MeshcatVisualizer = None
    ) -> None:
        if not settings.is_valid():
            raise ValueError("Settings are not valid.")
        self._logger = logging.getLogger("[hippopt::TerrainVisualizer]")
        self._settings = settings
        mesh_file = self._create_ground_mesh()
        self._create_ground_urdf(mesh_file)
        self._viz = MeshcatVisualizer() if viz is None else viz
        self._viz.load_model_from_file(
            model_path=os.path.join(
                self._settings.working_folder,
                self._settings.terrain.get_name() + ".urdf",
            ),
            model_name="terrain",
            color=self._settings.terrain_color,
        )
        if self._settings.draw_terrain_normals:
            self._draw_ground_normals()

    def _create_ground_urdf(self, mesh_filename: str) -> None:
        filename = self._settings.terrain.get_name() + ".urdf"
        full_filename = os.path.join(self._settings.working_folder, filename)
        if os.path.exists(full_filename):
            if self._settings.overwrite_terrain_files:
                self._logger.info(f"Overwriting {filename}")
            else:
                self._logger.info(f"{filename} already exists. Skipping creation.")
                return

        with open(full_filename, "w") as f:
            f.write(
                f"""<?xml version="1.0"?>
                <robot name="ground">
                    <link name="world"/>
                    <link name="link_0">
                        <inertial>
                            <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
                            <mass value="1.0"/>
                            <inertia ixx="0.0" ixy="0.0" ixz="0.0"
                                     iyy="0.0" iyz="0.0" izz="0.0"/>
                        </inertial>
                        <visual>
                            <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
                            <geometry>
                                <mesh filename="{mesh_filename}" scale="1.0 1.0 1.0"/>
                            </geometry>
                            <material name="ground_color">
                                 <color rgba="0.5 0.5 0.5 1"/>
                            </material>
                        </visual>
                        <collision>
                            <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
                            <geometry>
                                 <mesh filename="{mesh_filename}" scale="1.0 1.0 1.0"/>
                            </geometry>
                        </collision>
                    </link>
                    <joint name="joint_world_link_0" type="fixed">
                        <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
                        <parent link="world"/>
                        <child link="link_0"/>
                    </joint>
                </robot>
                """
            )

    def _create_ground_mesh(self) -> str:
        filename = self._settings.terrain.get_name() + ".stl"
        full_filename = os.path.join(self._settings.working_folder, filename)
        if os.path.exists(full_filename):
            if self._settings.overwrite_terrain_files:
                self._logger.info(f"Overwriting {filename}")
            else:
                self._logger.info(f"{filename} already exists. Skipping creation.")
                return full_filename
        x_step = (
            self._settings.terrain_x_limits[1] - self._settings.terrain_x_limits[0]
        ) / self._settings.terrain_mesh_axis_points
        y_step = (
            self._settings.terrain_y_limits[1] - self._settings.terrain_y_limits[0]
        ) / self._settings.terrain_mesh_axis_points
        x = np.arange(
            self._settings.terrain_x_limits[0],
            self._settings.terrain_x_limits[1] + x_step,
            x_step,
        )
        y = np.arange(
            self._settings.terrain_y_limits[0],
            self._settings.terrain_y_limits[1] + y_step,
            y_step,
        )
        x, y = np.meshgrid(x, y)
        assert x.shape == y.shape
        points = np.array([x.flatten(), y.flatten(), np.zeros(x.size)])
        height_function_map = self._settings.terrain.height_function().map(x.size)
        z = -np.array(height_function_map(points).full()).reshape(x.shape)
        surf2stl.write(full_filename, x, y, z)
        return full_filename

    def _draw_ground_normals(self) -> None:
        x_step = (
            self._settings.terrain_x_limits[1] - self._settings.terrain_x_limits[0]
        ) / self._settings.terrain_normal_axis_points
        y_step = (
            self._settings.terrain_y_limits[1] - self._settings.terrain_y_limits[0]
        ) / self._settings.terrain_normal_axis_points
        x = np.arange(
            self._settings.terrain_x_limits[0],
            self._settings.terrain_x_limits[1] + x_step,
            x_step,
        )
        y = np.arange(
            self._settings.terrain_y_limits[0],
            self._settings.terrain_y_limits[1] + y_step,
            y_step,
        )
        x, y = np.meshgrid(x, y)
        assert x.shape == y.shape
        points = np.array([x.flatten(), y.flatten(), np.zeros(x.size)])
        height_function_map = self._settings.terrain.height_function().map(x.size)
        z = -np.array(height_function_map(points).full()).reshape(x.shape)
        points = np.array([x.flatten(), y.flatten(), z.flatten()])
        normal_function_map = self._settings.terrain.normal_direction_function().map(
            x.size
        )
        normals = normal_function_map(points).full()

        for i in range(normals.shape[1]):
            self._viz.load_arrow(
                radius=self._settings.terrain_normals_radius,
                color=self._settings.terrain_normals_color,
                shape_name=f"normal_{i}",
            )
            self._viz.set_arrow_transform(
                origin=points[:, i],
                vector=self._settings.terrain_normal_scaling * normals[:, i],
                shape_name=f"normal_{i}",
            )
