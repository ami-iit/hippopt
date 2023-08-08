import copy
import dataclasses
import typing

import adam.casadi
import casadi as cs
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
    terrain: hp_rp.TerrainDescriptor = dataclasses.field(default=None)
    planar_dcc_height_multiplier: float = dataclasses.field(default=None)
    dcc_gain: float = dataclasses.field(default=None)
    dcc_epsilon: float = dataclasses.field(default=None)
    static_friction: float = dataclasses.field(default=None)
    maximum_velocity_control: np.ndarray = dataclasses.field(default=None)
    maximum_force_derivative: np.ndarray = dataclasses.field(default=None)
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
        self.terrain = hp_rp.PlanarTerrain()
        self.planar_dcc_height_multiplier = 10.0
        self.dcc_gain = 20.0
        self.dcc_epsilon = 0.05
        self.static_friction = 0.3
        self.maximum_velocity_control = np.ndarray([2.0, 2.0, 5.0])
        self.maximum_force_derivative = np.ndarray([100.0, 100.0, 100.0])

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
    centroidal_momentum_initial: hp.StorageType = hp.default_storage_field(hp.Parameter)

    dt: hp.StorageType = hp.default_storage_field(hp.Parameter)
    gravity: hp.StorageType = hp.default_storage_field(hp.Parameter)
    planar_dcc_height_multiplier: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )
    dcc_gain: hp.StorageType = hp.default_storage_field(hp.Parameter)
    dcc_epsilon: hp.StorageType = hp.default_storage_field(hp.Parameter)
    static_friction: hp.StorageType = hp.default_storage_field(hp.Parameter)
    maximum_velocity_control: hp.StorageType = hp.default_storage_field(hp.Parameter)
    maximum_force_derivative: hp.StorageType = hp.default_storage_field(hp.Parameter)

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

        self.com_initial = np.zeros(3)
        self.centroidal_momentum_initial = np.zeros(6)

        self.planar_dcc_height_multiplier = settings.planar_dcc_height_multiplier
        self.dcc_gain = settings.dcc_gain
        self.dcc_epsilon = settings.dcc_epsilon
        self.static_friction = settings.static_friction
        self.maximum_velocity_control = settings.maximum_velocity_control
        self.maximum_force_derivative = settings.maximum_force_derivative


class HumanoidWalkingFlatGround:
    def __init__(self, settings: Settings) -> None:
        if not settings.is_valid():
            raise ValueError("Settings are not valid")
        self.settings = copy.deepcopy(settings)
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

        function_inputs = {
            "mass_name": sym.mass.name(),
            "momentum_name": sym.centroidal_momentum.name(),
            "com_name": sym.com.name(),
            "gravity_name": sym.gravity.name(),
            "point_position_names": [],
            "point_force_names": [],
            "point_position_name": "p",
            "point_force_name": "f",
            "point_velocity_name": "v",
            "point_force_derivative_name": "f_dot",
            "point_position_control_name": "u_p",
            "height_multiplier_name": "kt",
            "dcc_gain_name": "k_bs",
            "dcc_epsilon_name": "eps",
            "static_friction_name": "mu_s",
            "options": self.settings.casadi_function_options,
        }

        self.settings.terrain.change_options(**function_inputs)

        dcc_planar_fun = hp_rp.dcc_planar_complementarity(
            terrain=self.settings.terrain,
            **function_inputs,
        )

        dcc_margin_fun = hp_rp.dcc_complementarity_margin(
            terrain=self.settings.terrain,
            **function_inputs,
        )

        friction_margin_fun = hp_rp.friction_cone_square_margin(
            terrain=self.settings.terrain, **function_inputs
        )

        height_fun = self.settings.terrain.height_function()
        normal_force_fun = hp_rp.normal_force_component(
            terrain=self.settings.terrain, **function_inputs
        )

        for point in sym.contact_points.left + sym.contact_points.right:
            problem.add_dynamics(
                hp.dot(point.f) == point.f_dot,
                x0=point.f0,
                integrator=default_integrator,
            )
            problem.add_dynamics(
                hp.dot(point.p) == point.v, x0=point.p0, integrator=default_integrator
            )

            dcc_planar = dcc_planar_fun(
                p=point.p, kt=sym.planar_dcc_height_multiplier, u_p=point.u_v
            )["planar_complementarity"]
            problem.add_expression_to_horizon(
                expression=cs.MX(point.v == dcc_planar), apply_to_first_elements=True
            )

            dcc_margin = dcc_margin_fun(
                p=point.p,
                f=point.f,
                v=point.v,
                f_dot=point.f_dot,
                k_bs=sym.dcc_gain,
                eps=sym.dcc_epsilon,
            )["dcc_complementarity_margin"]
            problem.add_expression_to_horizon(
                expression=cs.MX(dcc_margin >= 0), apply_to_first_elements=True
            )

            point_height = height_fun(p=point.p)["point_height"]
            problem.add_expression_to_horizon(
                expression=cs.MX(point_height >= 0), apply_to_first_elements=False
            )

            normal_force = normal_force_fun(p=point.p, f=point.f)["normal_force"]
            problem.add_expression_to_horizon(
                expression=cs.MX(normal_force >= 0), apply_to_first_elements=False
            )

            friction_margin = friction_margin_fun(
                p=point.p,
                f=point.f,
                mu_s=sym.static_friction,
            )["friction_cone_square_margin"]
            problem.add_expression_to_horizon(
                expression=cs.MX(friction_margin >= 0), apply_to_first_elements=False
            )

            problem.add_expression_to_horizon(
                expression=cs.Opti_bounded(
                    -sym.maximum_velocity_control,
                    point.u_v,
                    sym.maximum_velocity_control,
                ),
                apply_to_first_elements=True,
            )

            problem.add_expression_to_horizon(
                expression=cs.Opti_bounded(
                    -sym.maximum_force_control,
                    point.u_f,
                    sym.maximum_force_control,
                ),
                apply_to_first_elements=True,
            )

            function_inputs["point_position_names"].append(point.p.name())
            function_inputs["point_force_names"].append(point.f.name())

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

        com_dynamics = hp_rp.com_dynamics_from_momentum(**function_inputs)
        centroidal_dynamics = hp_rp.centroidal_dynamics_with_point_forces(
            **function_inputs
        )

        problem.add_dynamics(
            hp.dot(sym.com) == com_dynamics,
            x0=sym.com_initial,
            integrator=default_integrator,
        )

        problem.add_dynamics(
            hp.dot(sym.centroidal_momentum) == centroidal_dynamics,
            x0=sym.centroidal_momentum_initial,
            integrator=default_integrator,
        )

    def set_initial_conditions(self) -> None:  # TODO: fill
        pass
