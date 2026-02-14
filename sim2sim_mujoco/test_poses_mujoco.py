import mujoco
import mujoco.viewer
import time
import numpy as np

# --- CONFIGURATION ---
robot_model_path = "mujoco_xml/SF_biped.xml"

# The values you are currently using in the main script
# Are these creating a "Standing/Crouch" pose? Or something weird?
test_default_pos = np.array([0.3, 0.0, 0.6, -0.3, 0.0, -0.6])

joint_names_check = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_knee_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_knee_joint",
]


def main():
    print(f"Loading Model: {robot_model_path}")
    model = mujoco.MjModel.from_xml_path(robot_model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset
        mujoco.mj_resetData(model, data)

        # 1. PIN THE BASE (So we can see the pose clearly)
        data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        # 2. FORCE THE ROBOT TO "DEFAULT POSE"
        print("------------------------------------------------")
        print("Applying DEFAULT POSE:", test_default_pos)
        print("LOOK AT THE ROBOT:")
        print("1. Are the feet parallel?")
        print("2. Is it in a nice crouching/standing stance?")
        print("3. Or do the legs look crossed/broken?")
        print("------------------------------------------------")

        for i, name in enumerate(joint_names_check):
            addr = model.joint(name).qposadr
            data.qpos[addr] = test_default_pos[i]

            # Update the target too so it holds this pose
            data.ctrl[i] = test_default_pos[i]

        mujoco.mj_forward(model, data)

        while viewer.is_running():
            # Keep pinning the base
            data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
            data.qvel[0:6] = 0.0

            # Keep holding the pose
            data.ctrl[:] = test_default_pos

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
