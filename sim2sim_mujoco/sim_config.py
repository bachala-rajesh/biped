class Sim2simCfg:
    class sim_config:
        sim_dt = 1 / 200
        decimation = 4

    class robot_config:
        # observation
        obs_history_len = 5
        num_obs_terms = 8
        lin_vel_scale = 1.0
        ang_vel_scale = 1.0
        dof_pos_scale = 1.0
        dof_vel_scale = 1.0

        # joints
        joint_names = {
            "left_hip_pitch_joint": 0.3,
            "right_hip_pitch_joint": -0.3,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_knee_joint": 0.6,
            "right_knee_joint": -0.6,
        }
        initial_height = 0.53
        action_scale = 0.25

        # gait
        gait_freq = 1.75  # [Hz]
        gait_phase = 0.5  # [0-1]
        gait_duration = 0.5  # [0-1]

        # mujoco model gains
        stiffness_gain = 40.0
        damping_gain = -2.5
