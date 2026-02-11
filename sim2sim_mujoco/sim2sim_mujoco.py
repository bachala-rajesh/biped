import numpy as np
import mujoco
import mujoco.viewer
import time
import torch
from utils import HistoryBuffer, get_mujoco_data, get_projected_gravity


###########################################################
# configuration variables
###########################################################
rl_model_path = (
    "../logs/rsl_rl/bipedal_locomotion/2025-12-28_22-05-29_flat/exported/policy.pt"
)
# rl_model_path = "/home/mira/isaaclab_ws/biped/logs/rsl_rl/bipedal_locomotion/2025-12-29_02-22-56_rough/exported/policy.pt"

robot_model_path = "mujoco_xml/SF_biped.xml"

joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
]
initial_joint_pos = np.array([0.3, 0.0, 0.6, -0.3, 0.0, -0.6], dtype=np.float32)
initial_height = 0.53

sim_dt = 1 / 200
decimation = 4

cmd_vel = np.array([0.0, 0.0, 0.0])

# gait parameters
gait_freq = 2.0  # [Hz]
gait_phase = 0.5  # [0-1]
gait_duration = 0.5  # [0-1]
obs_gait_command = np.array([gait_freq, gait_phase, gait_duration], dtype=np.float32)

history_len = 5

# scales
action_scale = 0.25


class ObsScales:
    lin_vel = 1.0
    ang_vel = 1.0
    dof_pos = 1.0
    dof_vel = 0.05


# gains
stiffness_gain = 10.0
damping_gain = -2.5


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

    # form the observation vector
    current_obs = np.concatenate(
        [
            obs_ang_vel.reshape(-1),  # [:,3]
            obs_proj_gravity.reshape(-1),  # [:,3]
            obs_joint_pos.reshape(-1),  # [:,6]
            obs_joint_vel.reshape(-1),  # [:,6]
            obs_last_actions.reshape(-1),  # [:,6]
            obs_cmd.reshape(-1),  # [:,3]
            obs_gait_phase_sin_cos.reshape(-1),  # [:,2]
            obs_gait_command.reshape(-1),  # [:,3]
        ]
    )

    return current_obs


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
    # model.actuator_gainprm[:, 0] = stiffness_gain  # Stiffness
    # model.actuator_biasprm[:, 2] = damping_gain  # Damping

    # init history buffer
    history_buffer = HistoryBuffer(history_len=history_len)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # reset simulation
        mujoco.mj_resetData(model, data)

        # set initial joint positions
        for i, name in enumerate(joint_names):
            addr = model.joint(name).qposadr
            data.qpos[addr] = initial_joint_pos[i]
        data.qpos[2] = initial_height

        # forward pass
        mujoco.mj_forward(model, data)

        # initial buffer fill
        print("initializing history buffer")
        init_obs = get_observation(
            model, data, np.zeros(6, dtype=np.float32), 0.0, cmd_vel
        )
        for _ in range(history_len):
            history_buffer.update_history(init_obs)

        # loop variables
        step_counter = 0
        last_actions = np.zeros(6, dtype=np.float32)
        gait_time_accumulator = 0.0
        # real_start_time = time.time()

        print("------------------------------------------------")
        print("DROPPING ROBOT... Brain is OFF for 2 seconds.")
        print("------------------------------------------------")

        start_time = time.time()
        real_start_time = time.time()
        warmup_delay = 0.10  # Wait 2 seconds before turning on Policy

        while viewer.is_running():
            # camera settings
            # viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
            viewer.cam.distance = (
                6.0  # Zoom out (Increase this to widen the view further)
            )
            viewer.cam.azimuth = 135  # Rotate camera (0 = Behind, 90 = Right Side). 45 degree angle is usually best for depth
            viewer.cam.elevation = -20  # Look slightly down

            # Check if we are still in Warmup
            is_warmup = (time.time() - start_time) < warmup_delay

            # decimation loop
            if step_counter % decimation == 0:
                # update gait clock
                gait_time_accumulator += sim_dt * decimation

                # get observation
                current_obs = get_observation(
                    model, data, last_actions, gait_time_accumulator, cmd_vel
                )

                # update history buffer
                history_buffer.update_history(current_obs)
                stacked_obs = history_buffer.get_stacked_obs()

                # inference
                if is_warmup:
                    actions = np.zeros(6, dtype=np.float32)
                else:
                    obs_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).float()
                    with torch.no_grad():
                        actions = policy(obs_tensor)
                        actions = actions.detach().cpu().numpy().flatten()

                # update last actions
                # actions[2] *= -1.0
                # actions[5] *= -1.0
                actions = np.clip(actions, -100.0, 100.0)
                last_actions = actions

            # ---- physics step ----
            targets = (last_actions * action_scale) + initial_joint_pos
            data.ctrl[:] = targets

            # step simulation
            mujoco.mj_step(model, data)

            # update viewer
            viewer.sync()

            # update step counter
            step_counter += 1

            # Simple print to let you know when Brain turns ON
            if is_warmup and (time.time() - start_time) > warmup_delay - 0.1:
                print(">>> LANDED. Brain turning ON now! <<<")

            time_until_next_step = model.opt.timestep - (time.time() - real_start_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            real_start_time = time.time()


if __name__ == "__main__":
    main()


# ############ experiments
# 1. added the intial height
# 2. added the observation scaling
# 3. chnaged the damping and stiffness parameters
# 4. added the manual sign flipping for the actions (not working)
