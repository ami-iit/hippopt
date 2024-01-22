import copy
import dataclasses
import logging
import pathlib
import time

import ffmpeg
import liecasadi
import numpy as np
from idyntree.visualize import MeshcatVisualizer

from hippopt.robot_planning.utilities.terrain_visualizer import (
    TerrainVisualizer,
    TerrainVisualizerSettings,
)
from hippopt.robot_planning.variables.humanoid import (
    FeetContactPointDescriptors,
    HumanoidState,
)


@dataclasses.dataclass
class HumanoidStateVisualizerSettings(TerrainVisualizerSettings):
    robot_model: str = dataclasses.field(default=None)
    considered_joints: list[str] = dataclasses.field(default=None)
    contact_points: FeetContactPointDescriptors = dataclasses.field(default=None)
    robot_color: list[float] = dataclasses.field(default=None)
    com_color: list[float] = dataclasses.field(default=None)
    com_radius: float = dataclasses.field(default=None)
    contact_points_color: list[float] = dataclasses.field(default=None)
    contact_points_radius: float = dataclasses.field(default=None)
    contact_forces_color: list[float] = dataclasses.field(default=None)
    contact_force_radius: float = dataclasses.field(default=None)
    contact_force_scaling: float = dataclasses.field(default=None)
    pre_allocated_clones: int = dataclasses.field(default=None)

    def __post_init__(self):
        TerrainVisualizerSettings.__post_init__(self)
        if self.robot_color is None:
            self.robot_color = [1, 1, 1, 0.25]

        if self.com_color is None:
            self.com_color = [1, 0, 0, 1]

        if self.com_radius is None:
            self.com_radius = 0.02

        if self.contact_points_color is None:
            self.contact_points_color = [0, 0, 0, 1]

        if self.contact_forces_color is None:
            self.contact_forces_color = [1, 0, 0, 1]

        if self.contact_points_radius is None:
            self.contact_points_radius = 0.01

        if self.contact_force_radius is None:
            self.contact_force_radius = 0.005

        if self.contact_force_scaling is None:
            self.contact_force_scaling = 0.01

        if self.pre_allocated_clones is None:
            self.pre_allocated_clones = 1

    def is_valid(self) -> bool:
        ok = True
        logger = logging.getLogger("[hippopt::HumanoidStateVisualizerSettings]")
        if not TerrainVisualizerSettings.is_valid(self):
            ok = False
        if self.robot_model is None:
            logger.error("robot_model is not specified.")
            ok = False
        if self.considered_joints is None:
            logger.error("considered_joints is not specified.")
            ok = False
        if self.contact_points is None:
            logger.error("contact_points is not specified.")
            ok = False
        if len(self.robot_color) != 4:
            logger.error("robot_color is not specified correctly.")
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
        self._number_of_clones = self._settings.pre_allocated_clones
        self._viz = MeshcatVisualizer()
        self._terrain_visualizer = TerrainVisualizer(self._settings, self._viz)
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
            self._viz.load_arrow(
                radius=self._settings.contact_force_radius,
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
            ).flatten()

            self._viz.set_arrow_transform(
                origin=np.array(point.p).flatten(),
                vector=point_force,
                shape_name=f"[{index}]f_{i}",
            )

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

        for i in range(number_of_clones, len(states) + 1):
            initial_index = i - number_of_clones
            visualized_states = states[initial_index:i]
            if number_of_clones > 1:
                self._logger.info(
                    f"Visualizing states [{i-number_of_clones + 1},{i}]"
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
            sleep_time = _timestep_s[i - 1] * time_multiplier - elapsed_s
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
