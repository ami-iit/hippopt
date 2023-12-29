import copy
import dataclasses
import logging
import os
import pathlib
import time

import ffmpeg
import liecasadi
import numpy as np
from idyntree.visualize import MeshcatVisualizer

import hippopt.deps.surf2stl as surf2stl
from hippopt.robot_planning.utilities.terrain_descriptor import TerrainDescriptor
from hippopt.robot_planning.variables.humanoid import (
    FeetContactPointDescriptors,
    HumanoidState,
)


@dataclasses.dataclass
class HumanoidStateVisualizerSettings:
    robot_model: str = dataclasses.field(default=None)
    considered_joints: list[str] = dataclasses.field(default=None)
    contact_points: FeetContactPointDescriptors = dataclasses.field(default=None)
    robot_color: list[float] = dataclasses.field(default=None)
    ground_color: list[float] = dataclasses.field(default=None)
    terrain: TerrainDescriptor = dataclasses.field(default=None)
    com_color: list[float] = dataclasses.field(default=None)
    com_radius: float = dataclasses.field(default=None)
    contact_points_color: list[float] = dataclasses.field(default=None)
    contact_points_radius: float = dataclasses.field(default=None)
    contact_forces_color: list[float] = dataclasses.field(default=None)
    contact_force_radius: float = dataclasses.field(default=None)
    contact_force_scaling: float = dataclasses.field(default=None)
    working_folder: str = dataclasses.field(default=None)
    ground_mesh_axis_points: int = dataclasses.field(default=None)
    ground_x_limits: list[float] = dataclasses.field(default=None)
    ground_y_limits: list[float] = dataclasses.field(default=None)
    overwrite_ground_files: bool = dataclasses.field(default=None)
    pre_allocated_clones: int = dataclasses.field(default=None)

    def __post_init__(self):
        self.robot_color = [1, 1, 1, 0.25]
        self.ground_color = [0.5, 0.5, 0.5, 0.75]
        self.com_color = [1, 0, 0, 1]
        self.com_radius = 0.02
        self.contact_points_color = [0, 0, 0, 1]
        self.contact_forces_color = [1, 0, 0, 1]
        self.contact_points_radius = 0.01
        self.contact_force_radius = 0.005
        self.contact_force_scaling = 0.01
        self.ground_x_limits = [-1.5, 1.5]
        self.ground_y_limits = [-1.5, 1.5]
        self.ground_mesh_axis_points = 200
        self.working_folder = "./"
        self.overwrite_ground_files = False
        self.pre_allocated_clones = 1

    def is_valid(self) -> bool:
        ok = True
        logger = logging.getLogger("[hippopt::HumanoidStateVisualizerSettings]")
        if self.robot_model is None:
            logger.error("robot_model is not specified.")
            ok = False
        if self.considered_joints is None:
            logger.error("considered_joints is not specified.")
            ok = False
        if self.contact_points is None:
            logger.error("contact_points is not specified.")
            ok = False
        if not os.access(self.working_folder, os.W_OK):
            logger.error("working_folder is not writable.")
            ok = False
        if len(self.robot_color) != 4:
            logger.error("robot_color is not specified correctly.")
            ok = False
        if len(self.ground_color) != 4:
            logger.error("ground_color is not specified correctly.")
            ok = False
        if len(self.com_color) != 4:
            logger.error("com_color is not specified correctly.")
            ok = False
        if len(self.contact_points_color) != 4:
            logger.error("contact_points_color is not specified correctly.")
            ok = False
        if len(self.contact_forces_color) != 4:
            logger.error("contact_forces_color is not specified correctly.")
            ok = False
        if self.com_radius <= 0:
            logger.error("com_radius is not specified correctly.")
            ok = False
        if self.contact_points_radius <= 0:
            logger.error("contact_points_radius is not specified correctly.")
            ok = False
        if self.contact_force_radius <= 0:
            logger.error("contact_force_radius is not specified correctly.")
            ok = False
        if self.ground_mesh_axis_points <= 0:
            logger.error("ground_mesh_axis_points is not specified correctly.")
            ok = False
        if self.terrain is None:
            logger.error("terrain is not specified.")
            ok = False
        if len(self.ground_x_limits) != 2 or (
            self.ground_x_limits[0] >= self.ground_x_limits[1]
        ):
            logger.error("ground_x_limits are not specified correctly.")
            ok = False
        if len(self.ground_y_limits) != 2 or (
            self.ground_y_limits[0] >= self.ground_y_limits[1]
        ):
            logger.error("ground_y_limits are not specified correctly.")
            ok = False
        if self.contact_force_scaling <= 0:
            logger.error("contact_force_scaling is not specified correctly.")
            ok = False
        if self.pre_allocated_clones <= 0:
            logger.error("pre_allocated_clones is not specified correctly.")
            ok = False
        return ok


class HumanoidStateVisualizer:
    def __init__(self, settings: HumanoidStateVisualizerSettings) -> None:
        if not settings.is_valid():
            raise ValueError("Settings are not valid.")
        self._logger = logging.getLogger("[hippopt::HumanoidStateVisualizer]")
        self._settings = settings
        mesh_file = self._create_ground_mesh()
        self._create_ground_urdf(mesh_file)
        self._number_of_clones = self._settings.pre_allocated_clones
        self._viz = MeshcatVisualizer()
        self._viz.load_model_from_file(
            model_path=os.path.join(
                self._settings.working_folder,
                self._settings.terrain.get_name() + ".urdf",
            ),
            model_name="ground",
            color=self._settings.ground_color,
        )
        for i in range(self._number_of_clones):
            self._allocate_clone(i)
            if i != 0:
                self._set_clone_visibility(i, False)

    def _allocate_clone(self, index: int) -> None:
        self._viz.load_model_from_file(
            model_path=self._settings.robot_model,
            model_name=f"[{index}]robot",
            considered_joints=self._settings.considered_joints,
            color=self._settings.robot_color,
        )
        self._viz.load_sphere(
            radius=self._settings.com_radius,
            shape_name=f"[{index}]CoM",
            color=self._settings.com_color,
        )
        for i, point in enumerate(
            (self._settings.contact_points.left + self._settings.contact_points.right)
        ):
            self._viz.load_sphere(
                radius=self._settings.contact_points_radius,
                shape_name=f"[{index}]p_{i}",
                color=self._settings.contact_points_color,
            )
            self._viz.load_cylinder(
                radius=self._settings.contact_force_radius,
                length=1.0,
                shape_name=f"[{index}]f_{i}",
                color=self._settings.contact_forces_color,
            )

    def _set_clone_visibility(self, index: int, visible: bool) -> None:
        self._viz.viewer[f"[{index}]robot"].set_property(key="visible", value=visible)
        self._viz.viewer[f"[{index}]CoM"].set_property(key="visible", value=visible)
        for i, point in enumerate(
            (self._settings.contact_points.left + self._settings.contact_points.right)
        ):
            self._viz.viewer[f"[{index}]p_{i}"].set_property(
                key="visible", value=visible
            )
            self._viz.viewer[f"[{index}]f_{i}"].set_property(
                key="visible", value=visible
            )

    @staticmethod
    def _skew(x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0, -x[2], x[1]],
                [x[2], 0, -x[0]],
                [-x[1], x[0], 0],
            ]
        )

    def _get_force_scaled_rotation(self, point_force: np.ndarray) -> np.ndarray:
        force_norm = np.linalg.norm(point_force)
        scaling = np.diag([1, 1, force_norm])

        if force_norm < 1e-6:
            return scaling

        force_direction = point_force / force_norm
        cos_angle = np.dot(np.array([0, 0, 1]), force_direction)
        rotation_axis = self._skew(np.array([0, 0, 1])) @ force_direction

        if np.linalg.norm(rotation_axis) < 1e-6:
            return scaling

        skew_symmetric_matrix = self._skew(rotation_axis)
        rotation = (
            np.eye(3)
            + skew_symmetric_matrix
            + np.dot(skew_symmetric_matrix, skew_symmetric_matrix)
            * ((1 - cos_angle) / (np.linalg.norm(rotation_axis) ** 2))
        )
        return rotation @ scaling

    def _update_clone(self, index: int, state: HumanoidState) -> None:
        self._viz.set_multibody_system_state(
            np.array(state.kinematics.base.position).flatten(),
            liecasadi.SO3.from_quat(state.kinematics.base.quaternion_xyzw)
            .as_matrix()
            .full(),
            np.array(state.kinematics.joints.positions).flatten(),
            f"[{index}]robot",
        )
        self._viz.set_primitive_geometry_transform(
            np.array(state.com).flatten(),
            np.eye(3),
            f"[{index}]CoM",
        )

        for i, point in enumerate(
            (state.contact_points.left + state.contact_points.right)
        ):
            self._viz.set_primitive_geometry_transform(
                np.array(point.p).flatten(),
                np.eye(3),
                f"[{index}]p_{i}",
            )

            point_force = (
                np.array(point.f).flatten() * self._settings.contact_force_scaling
            )

            # Copied from https://github.com/robotology/idyntree/pull/1087 until it is
            # available in conda

            position = point.p + point_force / 2  # the origin is in the cylinder center
            transform = np.zeros((4, 4))
            transform[0:3, 3] = np.array(position).flatten()
            transform[3, 3] = 1
            transform[0:3, 0:3] = self._get_force_scaled_rotation(point_force)
            self._viz.viewer[f"[{index}]f_{i}"].set_transform(transform)

    def _create_ground_urdf(self, mesh_filename: str) -> None:
        filename = self._settings.terrain.get_name() + ".urdf"
        full_filename = os.path.join(self._settings.working_folder, filename)
        if os.path.exists(full_filename):
            if self._settings.overwrite_ground_files:
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
            if self._settings.overwrite_ground_files:
                self._logger.info(f"Overwriting {filename}")
            else:
                self._logger.info(f"{filename} already exists. Skipping creation.")
                return full_filename
        x_step = (
            self._settings.ground_x_limits[1] - self._settings.ground_x_limits[0]
        ) / self._settings.ground_mesh_axis_points
        y_step = (
            self._settings.ground_y_limits[1] - self._settings.ground_y_limits[0]
        ) / self._settings.ground_mesh_axis_points
        x = np.arange(
            self._settings.ground_x_limits[0],
            self._settings.ground_x_limits[1] + x_step,
            x_step,
        )
        y = np.arange(
            self._settings.ground_y_limits[0],
            self._settings.ground_y_limits[1] + y_step,
            y_step,
        )
        x, y = np.meshgrid(x, y)
        assert x.shape == y.shape
        points = np.array([x.flatten(), y.flatten(), np.zeros(x.size)])
        height_function_map = self._settings.terrain.height_function().map(x.size)
        z = -np.array(height_function_map(points).full()).reshape(x.shape)
        surf2stl.write(full_filename, x, y, z)
        return full_filename

    def _visualize_single_state(
        self,
        states: list[HumanoidState],
        save: bool,
        file_name_stem: str,
    ) -> None:
        if len(states) > self._number_of_clones:
            self._logger.warning(
                f"Number of states ({len(states)}) is greater than the number of "
                f"allocated clones ({self._number_of_clones}). "
                "Creating new clones."
            )
        for i in range(self._number_of_clones, len(states)):
            self._allocate_clone(i)
            self._set_clone_visibility(i, False)
            self._number_of_clones += 1

        for i in range(len(states)):
            self._update_clone(i, states[i])
            self._set_clone_visibility(i, True)

        for i in range(len(states), self._number_of_clones):
            self._set_clone_visibility(i, False)

        if save:
            image = self._viz.viewer.get_image()
            image.save(file_name_stem + ".png")

    def _visualize_multiple_states(
        self,
        states: list[HumanoidState],
        timestep_s: float | list[float] | np.ndarray,
        time_multiplier: float,
        number_of_clones: int,
        save: bool,
        file_name_stem: str,
    ) -> None:
        _timestep_s = copy.deepcopy(timestep_s)
        if (
            _timestep_s is None
            or isinstance(_timestep_s, float)
            or _timestep_s.size == 1
        ):
            single_step = _timestep_s if _timestep_s is not None else 0.0
            _timestep_s = [single_step] * len(states)

        if len(_timestep_s) != len(states):
            raise ValueError("timestep_s and states have different lengths.")

        folder_name = f"{self._settings.working_folder}/{file_name_stem}_frames"

        if save:
            pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
            self._logger.info(
                f"Saving visualization frames in {folder_name}. "
                "Make sure to have the visualizer open, "
                "otherwise the process will hang."
            )

        frame_prefix = "frame_"

        for i in range(number_of_clones, len(states)):
            initial_index = i - number_of_clones
            visualized_states = states[initial_index:i]
            if number_of_clones > 1:
                self._logger.info(
                    f"Visualizing states [{i-number_of_clones + 1},{i + 1}]"
                    f" of {len(states)}."
                )
            else:
                self._logger.info(f"Visualizing state {i}/{len(states)}")
            start = time.time()
            self._visualize_single_state(
                visualized_states,
                save=save,
                file_name_stem=f"{folder_name}/{frame_prefix}{i-number_of_clones:03}",
            )
            end = time.time()
            elapsed_s = end - start
            sleep_time = _timestep_s[i] * time_multiplier - elapsed_s
            time.sleep(max(0.0, sleep_time))

        if save:
            if timestep_s is None:
                self._logger.warning("timestep_s is None. Saving video with 1.0 fps.")
                fps = 1.0
            elif isinstance(timestep_s, list):
                if len(timestep_s) > 1:
                    self._logger.warning(
                        "The input timestep is a list. "
                        "Using the average to compute the fps."
                    )
                    fps = 1.0 / (sum(timestep_s) / len(timestep_s))
                else:
                    fps = 1.0 / timestep_s[0]
            elif isinstance(timestep_s, np.ndarray):
                if timestep_s.size > 1:
                    self._logger.warning(
                        "The input timestep is a list. "
                        "Using the average to compute the fps."
                    )
                    fps = 1.0 / (sum(timestep_s) / len(timestep_s))
                else:
                    fps = 1.0 / timestep_s
            elif isinstance(timestep_s, float):
                fps = 1.0 / timestep_s
            else:
                self._logger.warning("Using the fps=1.0")
                fps = 1.0

            self.generate_video_from_frames(
                file_name_stem=file_name_stem,
                frames_folder=folder_name,
                frames_prefix=frame_prefix,
                fps=fps / time_multiplier,
            )

    def generate_video_from_frames(
        self, file_name_stem: str, frames_folder: str, frames_prefix: str, fps: float
    ) -> None:
        frames = ffmpeg.input(
            filename=f"{frames_folder}/{frames_prefix}%3d.png",
            framerate=fps,
        )
        video = ffmpeg.output(
            frames,
            f"{self._settings.working_folder}/{file_name_stem}.mp4",
            video_bitrate="20M",
        )
        try:
            ffmpeg.run(video)
        except ffmpeg.Error as e:
            self._logger.error(
                "ffmpeg failed to generate the video. "
                "The following output might contain additional information: "
                + str(e)
                + " stderr: "
                + str(e.stderr)
                + " stdout: "
                + str(e.stdout)
            )

    def visualize(
        self,
        states: HumanoidState | list[HumanoidState],
        timestep_s: float | list[float] | np.ndarray = None,
        time_multiplier: float = 1.0,
        number_of_clones: int = 1,
        save: bool = False,
        file_name_stem: str = "humanoid_state_visualization",
    ) -> None:
        if number_of_clones < 1:
            raise ValueError("number_of_clones must be greater than 0.")

        if not isinstance(states, list):
            states = [states]

        if number_of_clones < len(states):
            self._visualize_multiple_states(
                states=states,
                timestep_s=timestep_s,
                time_multiplier=time_multiplier,
                number_of_clones=number_of_clones,
                save=save,
                file_name_stem=file_name_stem,
            )
        else:
            self._visualize_single_state(
                states=states, save=save, file_name_stem=file_name_stem
            )
