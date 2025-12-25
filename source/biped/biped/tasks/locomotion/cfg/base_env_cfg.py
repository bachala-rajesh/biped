# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


from biped.tasks.locomotion import mdp
from biped.tasks.locomotion.cfg.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    STAIRS_TERRAINS_CFG,
)
from biped.assets.config.simple_biped_config import BIPED_CONFIG

##
# Scene definition
##

@configclass
class BipedSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
   
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    
    # robot
    robot: ArticulationCfg = MISSING

    height_scanner: RayCasterCfg = MISSING

    # robot
    # robot = BIPED_CONFIG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
        
    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True, update_period=0.0
    )


##
# mdp components
##

@configclass
class CommandsCfg:
    """Command terms for the MDP"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.3,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        debug_vis=True,
        resampling_time_range=(5.0, 15.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.8), 
            lin_vel_y=(-0.2, 0.2), 
            ang_vel_z=(-math.pi, math.pi), 
            heading=(-0.5, 0.5)     #Limited turning- 29 degrees
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*joint"],
        scale=0.25,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025))

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=0.01))

        # last action
        last_action = ObsTerm(func=mdp.last_action)

        # velocity command
        vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # heights scan
        heights: ObsTerm = MISSING
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # # reset
    # reset_cart_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    #         "position_range": (-1.0, 1.0),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task Rewards --
    
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25}, # std 0.25 comes from 'tracking_sigma' in legacy
    )
    
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    #### custom rewards
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=8.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*knee_link"), 
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # -- Penalties --
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-4.5e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*hip.*", "base_link"]), 
            "threshold": 0.1
        },
    )

    # action_rate = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    #### custom penalities
    base_height = RewTerm(
        func=mdp.base_com_height,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "target_height": 0.49 # Kept your specific robot height
        }, 
    )
    
    # action_smoothness = RewTerm(
    #     func=mdp.ActionSmoothnessPenalty, 
    #     weight=-0.01, # Start small
    # )
    
    feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-0.1, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*knee_link"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # The Symmetry Check
    unbalance_feet = RewTerm(
        func=mdp.unbalance_feet_air_time,
        weight=-0.5, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*knee_link"),
        },
    )
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # # (1) Time out
    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )




##
# Environment configuration
##


@configclass
class BipedEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: BipedSceneCfg = BipedSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        
        
