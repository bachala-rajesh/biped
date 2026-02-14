"""Script for sim2sim deployment of a trained RLpolicy in Mujoco environment"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import torch
from utils import HistoryBuffer, get_mujoco_data, get_projected_gravity


# path variables
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
relative_policy_path = (
    "logs/rsl_rl/bipedal_locomotion/2026-02-12_08-33-37_flat/exported/policy.pt"
)
rl_model_path = os.path.join(project_root, relative_policy_path)
robot_model_path = os.path.join(script_dir, "mujoco_xml", "SF_biped.xml")

#  joint names and its initial position in the order the policy sees them.
joint_names = {
    "left_hip_pitch_joint": 0.3,
    "right_hip_pitch_joint": -0.3,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_knee_joint": 0.6,
    "right_knee_joint": -0.6,
}

initial_joint_pos = np.array(
    [joint_names[name] for name in joint_names.keys()], dtype=np.float32
)
initial_height = 0.53

# simulation related variables
sim_dt = 1 / 200
decimation = 4
history_len = 5
num_obs_terms = 8
action_scale = 0.25

# gait parameters
gait_freq = 1.75  # [Hz]
gait_phase = 0.5  # [0-1]
gait_duration = 0.5  # [0-1]
obs_gait_command = np.array([gait_freq, gait_phase, gait_duration], dtype=np.float32)

# command velocity
cmd_vel = np.array([0.0, 0.0, 0.0])

# mujoco model gains
stiffness_gain = 40.0
damping_gain = -2.5


# observation scales
class ObsScales:
    lin_vel = 1.0
    ang_vel = 1.0
    dof_pos = 1.0
    dof_vel = 1.0


def get_observation(model, data, last_actions, gait_time, cmd_vel):
    quat, ang_vel, joints_pos, joints_vel = get_mujoco_data(model, data, joint_names)
    obs_proj_gravity = get_projected_gravity(quat)

    # calculate gait phase
    gait_phase_val = (gait_time * gait_freq) % 1.0
    obs_gait_phase_sin_cos = np.array(
        [
            np.sin(2 * np.pi * gait_phase_val),
            np.cos(2 * np.pi * gait_phase_val),
        ],
        dtype=np.float32,
    )

    # apply scaling to observations
    obs_ang_vel = ang_vel * ObsScales.ang_vel
    relative_joint_pos = joints_pos - initial_joint_pos
    obs_joint_pos = relative_joint_pos * ObsScales.dof_pos
    obs_joint_vel = joints_vel * ObsScales.dof_vel
    obs_cmd = cmd_vel * np.array(
        [ObsScales.lin_vel, ObsScales.lin_vel, ObsScales.ang_vel],
        dtype=np.float32,
    )
    obs_last_actions = last_actions

    # form the observation vector as a list of numpy arrays
    current_obs = [
        obs_ang_vel.reshape(-1),  # [:,3]
        obs_proj_gravity.reshape(-1),  # [:,3]
        obs_cmd.reshape(-1),  # [:,3]
        obs_joint_pos.reshape(-1),  # [:,6]
        obs_joint_vel.reshape(-1),  # [:,6]
        obs_last_actions.reshape(-1),  # [:,6]
        obs_gait_phase_sin_cos.reshape(-1),  # [:,2]
        obs_gait_command.reshape(-1),  # [:,3]
    ]

    fall_status = False
    if abs(obs_proj_gravity[0]) > 0.90:
        fall_status = True

    return current_obs, fall_status


def main():
    # load policy
    print(f"Loading Policy: {rl_model_path}")
    policy = torch.jit.load(rl_model_path)
    policy.eval()

    # load model
    print(f"Loading Model: {robot_model_path}")
    model = mujoco.MjModel.from_xml_path(robot_model_path)
    data = mujoco.MjData(model)

    # set gains
    model.actuator_gainprm[:, 0] = stiffness_gain  # Stiffness
    model.actuator_biasprm[:, 2] = damping_gain  # Damping

    # init history buffer
    history_buffer = HistoryBuffer(history_len=history_len, num_obs_terms=num_obs_terms)

    # loop variables
    step_counter = 0
    last_actions = np.zeros(6, dtype=np.float32)
    gait_time_accumulator = 0.0

    #  time related variables
    start_time = time.time()
    real_start_time = time.time()
    warmup_delay = 0.10  # Wait few seconds before turning on Policy

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # reset simulation
        mujoco.mj_resetData(model, data)

        # set initial joint positions and height
        for i, name in enumerate(joint_names):
            addr = model.joint(name).qposadr
            data.qpos[addr] = initial_joint_pos[i]
        data.qpos[2] = initial_height

        # forward pass and inital buffer fill
        mujoco.mj_forward(model, data)
        fall_status = False
        init_obs_list, fall_status = get_observation(
            model, data, np.zeros(6, dtype=np.float32), 0.0, cmd_vel
        )
        for _ in range(history_len):
            history_buffer.update_history(init_obs_list)

        # camera settings
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 10.0  # Zoom out
        viewer.cam.azimuth = 135  # Rotate camera (0 = Behind, 90 = Right Side).
        viewer.cam.elevation = -20  # Look slightly down

        while viewer.is_running():
            # Check if we are still in Warmup
            is_warmup = (time.time() - start_time) < warmup_delay

            # decimation loop
            if step_counter % decimation == 0:
                # update gait clock
                gait_time_accumulator += sim_dt * decimation

                # get observation
                current_obs_list, fall_status = get_observation(
                    model, data, last_actions, gait_time_accumulator, cmd_vel
                )

                # update history buffer
                history_buffer.update_history(current_obs_list)
                stacked_obs = history_buffer.get_stacked_obs()

                # wait for warmup and then perform inference
                if is_warmup:
                    actions = np.zeros(6, dtype=np.float32)
                else:
                    obs_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).float()
                    with torch.no_grad():
                        actions = policy(obs_tensor)
                        actions = actions.detach().cpu().numpy().flatten()

                # update last actions
                actions = np.clip(actions, -100.0, 100.0)
                last_actions = actions

            # check if the robot has fallen
            if fall_status:
                print("Robot has fallen down...Exiting")
                break

            # ---- physics step ----
            targets = (last_actions * action_scale) + initial_joint_pos
            data.ctrl[:] = targets

            # step simulation
            mujoco.mj_step(model, data)

            # update viewer
            viewer.sync()

            # update step counter
            step_counter += 1

            time_until_next_step = model.opt.timestep - (time.time() - real_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            real_start_time = time.time()


if __name__ == "__main__":
    main()
