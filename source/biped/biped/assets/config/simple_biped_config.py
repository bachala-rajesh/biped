import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


current_dir = os.path.dirname(__file__)
usd_path = os.path.join(current_dir, "../usd/SF_bipedal_usd/SF_bipedal.usd")

BIPED_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=2.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=8,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_knee_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_knee_joint": 0.0,
        },
    ),
    actuators={
        "hip_pitch_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_pitch.*"],
            effort_limit=20.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=4.0,
        ),
        "hip_roll_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_roll.*"],
            effort_limit=20.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=4.0,
        ),
        "knee_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*knee.*"],
            effort_limit=20.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=4.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
