# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""script to debug the policy of an RL agent trained with RSL-RL library.
Script will print the observation and action terms in the order the policy sees them.
Script has also the option to view the observation data.

How to use:

for only printing the observation and action terms:
    python debug_policy.py   --task=biped_walk_flat_play      --checkpoint "exported_policy.pt"
for printing the observation and action terms and the observation data:
    python debug_policy.py   --task=biped_walk_flat_play      --checkpoint "exported_policy.pt" --obs_data_view
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# ----------------------------------------------------------#
# Observation and action variables
# ----------------------------------------------------------#
observation_term = [
    "AngVel",
    "Gravity",
    "Cmd",
    "J_Pos",
    "J_Vel",
    "Action",
    "Phase",
    "GaitCmd",
]
history_len = 5
term_shapes = [3, 3, 3, 6, 6, 6, 2, 3]
term_history_shapes = [s * history_len for s in term_shapes]


# ----------------------------------------------------------#
# Argparse arguments
# ----------------------------------------------------------#
# add argparse arguments
parser = argparse.ArgumentParser(description="debug policy of an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--obs_data_view",
    action="store_true",
    default=False,
    help="view the observation data.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ----------------------------------------------------------#
# Policy
# ----------------------------------------------------------#

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import biped.tasks  # noqa: F401
import numpy as np


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1  # set number of environments to 1 for debugging

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()


    # print the observation keys
    print("=" * 40)
    print(
        f"Policy observation terms: {env.unwrapped.observation_manager.active_terms['policy']}"
    )
    print(
        f"shape of policy observation: {env.unwrapped.observation_manager.group_obs_dim['policy']}"
    )
    print(
        f"shape of individual observation: {env.unwrapped.observation_manager.group_obs_term_dim['policy']}"
    )
    print(
        f"IO descriptors: {env.unwrapped.observation_manager.get_IO_descriptors['policy']}"
    )

    # print the action info
    print("=" * 40)
    print(f"action dimensions: {env.unwrapped.action_manager.total_action_dim}")
    print(f"Action shape: {env.unwrapped.action_manager.action_term_dim}")
    print(f"Action terms: {env.unwrapped.action_manager.get_IO_descriptors}")
    print("=" * 40 + "\n")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # ---------------- DEBUG PRINT ----------------
            if args_cli.obs_data_view:
                print("\n" + "=" * 40)
                current_ptr = 0
                obs_flat = obs[0]["policy"].cpu().numpy()
                for name, total_size in zip(observation_term, term_history_shapes):
                    # Calculate the size of a single time-step for a term
                    single_step_dim = total_size // history_len

                    #  Slice the full history block for a term
                    #  This contains [ t-4, t-3, t-2, t-1, t ]
                    term_block = obs_flat[current_ptr : current_ptr + total_size]

                    #  Extract ONLY the newest frame (the tail of the block)
                    latest_data = term_block[-single_step_dim:]

                    #  Print
                    data_str = np.array2string(
                        latest_data, precision=4, suppress_small=True, separator=", "
                    )
                    print(f"{name:12} [{single_step_dim}]: {data_str}")

                    #  Advance the pointer to the next term's block
                    current_ptr += total_size
                print("-" * 40)
                print(f"POLICY OUTPUT: {actions[0].cpu().numpy()}")
                print("-" * 40)
            # ----------------------------------------------------------#

            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
