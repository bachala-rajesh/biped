from isaaclab.utils import configclass

from biped.tasks.locomotion.cfg.base_env_cfg import BipedEnvCfg
from biped.assets.config.simple_biped_config import BIPED_CONFIG
from biped.tasks.locomotion.cfg.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)


@configclass
class BipedBaseEnvCfg(BipedEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BIPED_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "left_hip_pitch_joint": 0.3,
            "left_hip_roll_joint": 0.0,
            "left_knee_joint": 0.6,
            "right_hip_pitch_joint": -0.3,
            "right_hip_roll_joint": 0.0,
            "right_knee_joint": -0.6,
        }
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.53)

        # self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        # self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera
        # self.viewer.origin_type = "env"


@configclass
class BipedBaseEnvCfg_PLAY(BipedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # # disable randomization for play
        # self.observations.policy.enable_corruption = False
        # # remove random pushing event
        # self.events.push_robot = None
        # # remove random base mass addition event
        # self.events.add_base_mass = None


############################
# Biped Blind Flat Environment
############################


@configclass
class BipedBlindFlatEnvCfg(BipedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        # self.observations.critic.heights = None

        # self.curriculum.terrain_levels = None


@configclass
class BipedBlindFlatEnvCfg_PLAY(BipedBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        # self.observations.critic.heights = None

        # self.curriculum.terrain_levels = None


#############################
# Biped Blind Rough Environment
#############################


@configclass
class BipedBlindRoughEnvCfg(BipedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        # self.observations.critic.heights = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class BipedBlindRoughEnvCfg_PLAY(BipedBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        # self.observations.critic.heights = None

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG
