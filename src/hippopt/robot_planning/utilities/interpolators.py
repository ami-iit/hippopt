import casadi as cs
import liecasadi
import numpy as np

from hippopt import StorageType
from hippopt.robot_planning.variables.contacts import (
    ContactPointDescriptor,
    FootContactPhaseDescriptor,
    FootContactState,
)


def linear_interpolator(
    initial: StorageType, final: StorageType, number_of_points: int
) -> list[StorageType]:
    assert not isinstance(initial, list) and not isinstance(final, list)

    interpolator = cs.interpolant("lerp", "linear", [initial, final], [0.0, 1.0])
    x = np.linspace(start=0.0, stop=1.0, num=number_of_points)
    return [interpolator(x_i) for x_i in x]


def quaternion_slerp(
    initial: StorageType, final: StorageType, number_of_points: int
) -> list[StorageType]:
    assert not isinstance(initial, list) and not isinstance(final, list)

    x = np.linspace(start=0.0, stop=1.0, num=number_of_points)
    return [liecasadi.Quaternion.slerp_step(initial, final, t) for t in x]


def transform_interpolator(
    initial: liecasadi.SE3, final: liecasadi.SE3, number_of_points: int
) -> list[liecasadi.SE3]:
    linear_interpolation = linear_interpolator(
        initial=initial.translation(),
        final=final.translation(),
        number_of_points=number_of_points,
    )
    quaternion_interpolation = quaternion_slerp(
        initial=initial.rotation(),
        final=final.rotation(),
        number_of_points=number_of_points,
    )
    output = []
    for i in range(number_of_points):
        output.append(
            liecasadi.SE3(quaternion_interpolation[i], linear_interpolation[i])
        )
    return output


def foot_contact_state_interpolator(
    phases: list[FootContactPhaseDescriptor],
    descriptor: list[ContactPointDescriptor],
    number_of_points: int,
    dt: float,
    t0: float = 0.0,
) -> list[FootContactState]:
    assert len(phases) > 0
    assert number_of_points > 0
    assert dt > 0.0

    end_time = t0 + dt * number_of_points

    if phases[0].activation_time is None:
        deactivation_time = (
            phases[0].deactivation_time
            if phases[0].deactivation_time is not None
            else t0
        )
        phases[0].activation_time = min(deactivation_time, t0) - dt

    for i, phase in enumerate(phases):
        if phase.activation_time is None:
            raise ValueError(
                f"Phase {i} has no activation time, but is not the first phase."
            )

    last = len(phases) - 1
    if phases[last].deactivation_time is None:
        phases[last].deactivation_time = (
            max(end_time, phases[last].activation_time) + dt
        )

    if phases[last].deactivation_time < end_time:
        raise ValueError(
            f"The Last phase deactivation time "
            f"({phases[len(phases) - 1].deactivation_time}) is before "
            f"the end time ({end_time}, computed from the inputs)."
        )

    for i, phase in enumerate(phases):
        if phase.deactivation_time is None:
            raise ValueError(
                f"Phase {i} has no deactivation time, but is not the last phase."
            )
        if phase.activation_time > phase.deactivation_time:
            raise ValueError(
                f"Phase {i} has an activation time ({phase.activation_time}) "
                f"greater than its deactivation time ({phase.deactivation_time})."
            )

        if i < last:
            if phase.deactivation_time > phases[i + 1].activation_time:
                raise ValueError(
                    f"Phase {i} has a deactivation time ({phase.deactivation_time}) "
                    f"greater than the activation time of the next phase "
                    f"({phases[i + 1].activation_time})."
                )

    output = []

    def append_stance_phase(
        stance_phase: FootContactPhaseDescriptor,
        points: int,
    ) -> None:
        for _ in range(points):
            foot_state = FootContactState.from_parent_frame_transform(
                descriptor=descriptor, transform=stance_phase.transform
            )
            for point in foot_state:
                point.f = stance_phase.force
            output.append(foot_state)

    def append_swing_phase(
        start_phase: FootContactPhaseDescriptor,
        end_phase: FootContactPhaseDescriptor,
        points: int,
    ):
        full_swing_points = int(
            np.ceil((end_phase.activation_time - start_phase.deactivation_time) / dt)
        )
        mid_swing_points = min(round(full_swing_points / 2), points)
        mid_swing_transforms = transform_interpolator(
            start_phase.transform, start_phase.mid_swing_transform, mid_swing_points
        )
        for transform in mid_swing_transforms:
            foot_state = FootContactState.from_parent_frame_transform(
                descriptor=descriptor, transform=transform
            )
            for point in foot_state:
                point.f = 0.0
            output.append(foot_state)
        second_half_points = points - mid_swing_points
        if second_half_points == 0:
            return
        second_half_transforms = transform_interpolator(
            start_phase.mid_swing_transform, end_phase.transform, second_half_points
        )
        for transform in second_half_transforms:
            foot_state = FootContactState.from_parent_frame_transform(
                descriptor=descriptor, transform=transform
            )
            for point in foot_state:
                point.f = end_phase.force
            output.append(foot_state)

    if len(phases) == 1 or phases[0].deactivation_time >= end_time:
        append_stance_phase(phases[0], number_of_points)
        return output

    remaining_points = number_of_points
    for i in range(len(phases) - 1):
        phase = phases[i]
        next_phase = phases[i + 1]

        stance_points = int(
            np.ceil((phase.deactivation_time - phase.activation_time) / dt)
        )
        stance_points = min(stance_points, remaining_points)

        append_stance_phase(phase, stance_points)
        remaining_points -= stance_points

        if remaining_points == 0:
            return output

        swing_points = int(
            np.ceil((next_phase.activation_time - phase.deactivation_time) / dt)
        )

        swing_points = min(swing_points, remaining_points)

        if swing_points == 0:
            continue

        append_swing_phase(phase, next_phase, swing_points)
        remaining_points -= swing_points

        if remaining_points == 0:
            return output

    last_phase = phases[len(phases) - 1]
    append_stance_phase(last_phase, remaining_points)
    return output
