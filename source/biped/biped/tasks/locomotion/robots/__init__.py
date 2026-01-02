# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .. import agents

##
# Register Gym environments.
##

from .biped_env_cfg import (
    BipedBlindFlatEnvCfg,
    BipedBlindFlatEnvCfg_PLAY,
    BipedBlindRoughEnvCfg,
    BipedBlindRoughEnvCfg_PLAY,
    BipedBlindStairEnvCfg,
    BipedBlindStairEnvCfg_PLAY,
)


######################################
#### Biped blind flat environment
######################################


gym.register(
    id="biped_walk_flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BipedBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PointFootPPORunnerCfg",
    },
)

gym.register(
    id="biped_walk_flat_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BipedBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PointFootPPORunnerCfg",
    },
)




######################################
#### Biped blind rough environment
######################################


gym.register(
    id="biped_walk_rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BipedBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PointFootPPORunnerCfg",
    },
)

gym.register(
    id="biped_walk_rough_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BipedBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PointFootPPORunnerCfg",
    },
)



######################################
#### Biped blind stairs environment
######################################


gym.register(
    id="biped_walk_stairs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BipedBlindStairEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PointFootPPORunnerCfg",
    },
)

gym.register(
    id="biped_walk_stairs_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BipedBlindStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PointFootPPORunnerCfg",
    },
)
