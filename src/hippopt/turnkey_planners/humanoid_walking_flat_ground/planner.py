import dataclasses
import typing

import adam.casadi
import numpy as np

import hippopt as hp
import hippopt.integrators as hp_int
import hippopt.robot_planning as hp_rp


@dataclasses.dataclass
class ExtendedContactPoint(hp_rp.ContactPoint):
    u_v: hp.StorageType = hp.default_storage_field(hp.Variable)

    def __post_init__(self, input_descriptor: hp_rp.ContactPointDescriptor) -> None:
        super().__post_init__(input_descriptor)
        self.u_v = np.zeros(3)


@dataclasses.dataclass
class FeetContactPoints(hp.OptimizationObject):
    left: list[ExtendedContactPoint] = hp.default_composite_field()
    right: list[ExtendedContactPoint] = hp.default_composite_field()


@dataclasses.dataclass
class FeetContactPointDescriptors:
    left: list[hp_rp.ContactPointDescriptor] = dataclasses.field(default_factory=list)
    right: list[hp_rp.ContactPointDescriptor] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Settings:
    robot_urdf: str = dataclasses.field(default=None)
    joints_name_list: list[str] = dataclasses.field(default=None)
    contact_points: FeetContactPointDescriptors = dataclasses.field(default=None)
    root_link: str = dataclasses.field(default=None)
    gravity: np.array = dataclasses.field(default=None)
    horizon_length: int = dataclasses.field(default=None)
    integrator: typing.Type[hp.SingleStepIntegrator] = dataclasses.field(default=None)
    casadi_function_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.casadi_function_options = (
            self.casadi_function_options
            if isinstance(self.casadi_function_options, dict)
            else {}
        )
        self.root_link = "root_link"
        self.gravity = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])
        self.integrator = hp_int.ImplicitTrapezoid

    def is_valid(self) -> bool:
        return (
            self.robot_urdf is not None
            and self.joints_name_list is not None
            and self.contact_points is not None
            and self.horizon_length is not None
        )


@dataclasses.dataclass
class Variables(hp.OptimizationObject):
    contact_points: FeetContactPoints | list[
        FeetContactPoints
    ] = hp.default_composite_field()
    com: hp.StorageType = hp.default_storage_field(hp.Variable)
    centroidal_momentum: hp.StorageType = hp.default_storage_field(hp.Variable)
    mass: hp.StorageType = hp.default_storage_field(hp.Parameter)
    kinematics: hp_rp.FloatingBaseSystem = hp.default_composite_field()

    com_initial: hp.StorageType = hp.default_storage_field(hp.Parameter)

    dt: hp.StorageType = hp.default_storage_field(hp.Parameter)
    gravity: hp.StorageType = hp.default_storage_field(hp.Parameter)

    settings: dataclasses.InitVar[Settings] = dataclasses.field(default=None)
    kin_dyn_object: dataclasses.InitVar[
        adam.casadi.KinDynComputations
    ] = dataclasses.field(default=None)

    def __post_init__(
        self,
        settings: Settings,
        kin_dyn_object: adam.casadi.KinDynComputations,
    ) -> None:
        self.contact_points.left = [
            hp_rp.ContactPoint(descriptor=point)
            for point in settings.contact_points.left
        ]
        self.contact_points.right = [
            hp_rp.ContactPoint(descriptor=point)
            for point in settings.contact_points.right
        ]

        self.com = np.zeros(3)
        self.centroidal_momentum = np.zeros(6)
        self.kinematics = hp_rp.FloatingBaseSystem(kin_dyn_object.NDoF)
        self.dt = 0.1
        self.gravity = kin_dyn_object.g[:, 3]
        self.mass = kin_dyn_object.get_total_mass()


class HumanoidWalkingFlatGround:
    def __init__(self, settings: Settings) -> None:
        if not settings.is_valid():
            raise ValueError("Settings are not valid")
        self.settings = settings
        self.kin_dyn_object = adam.casadi.KinDynComputations(
            urdfstring=self.settings.robot_urdf,
            joints_name_list=self.settings.joints_name_list,
            root_link=self.settings.root_link,
            gravity=self.settings.gravity,
            f_opts=self.settings.casadi_function_options,
        )

        self.variables = Variables(
            settings=self.settings, kin_dyn_object=self.kin_dyn_object
        )

        self.ocp = hp.OptimalControlProblem.create(
            self.variables, horizon_length=self.settings.horizon_length
        )

        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure

        default_integrator = self.settings.integrator

        for point in sym.contact_points.left + sym.contact_points.right:
            problem.add_dynamics(
                hp.dot(point.force) == point.f_dot,
                x0=point.f0,
                integrator=default_integrator,
            )
            problem.add_dynamics(
                hp.dot(point.p) == point.v, x0=point.p0, integrator=default_integrator
            )

        problem.add_dynamics(
            hp.dot(sym.kinematics.base.position) == sym.kinematics.base.linear_velocity,
            x0=sym.kinematics.base.initial_position,
            integrator=default_integrator,
        )

        problem.add_dynamics(
            hp.dot(sym.kinematics.base.quaternion_xyzw)
            == sym.kinematics.base.quaternion_velocity_xyzw,
            x0=sym.kinematics.base.initial_quaternion_xyzw,
            integrator=default_integrator,
        )

        problem.add_dynamics(
            hp.dot(sym.kinematics.joints.positions) == sym.kinematics.joints.velocities,
            x0=sym.kinematics.joints.initial_positions,
            integrator=default_integrator,
        )

        problem.add_dynamics(
            hp.dot(sym.com) == sym.centroidal_momentum[:3] / sym.mass,
            x0=sym.com_initial,
            integrator=default_integrator,
        )

    def set_initial_conditions(self) -> None:  # TODO: fill
        pass
