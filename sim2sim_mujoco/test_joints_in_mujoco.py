import mujoco
import mujoco.viewer
import time
import numpy as np

# --- CONFIGURATION ---
robot_model_path = "mujoco_xml/SF_biped.xml"

# The mapping we want to verify
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
        # --- CAMERA CONFIGURATION ---
        # 1. Look at the robot (it's pinned at z=1.0)
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        # 2. Zoom out (Increase this to widen the view further)
        viewer.cam.distance = 4.0

        # 3. Rotate camera (0 = Behind, 90 = Right Side)
        viewer.cam.azimuth = 135  # 45 degree angle is usually best for depth
        viewer.cam.elevation = -20  # Look slightly down

        # Reset
        mujoco.mj_resetData(model, data)

        print("------------------------------------------------")
        print("PHASE 1: Robot is PINNED in mid-air.")
        print("We will move ONE joint at a time.")
        print("------------------------------------------------")

        # --- TEST SEQUENCE ---
        # 1. Test HIP PITCH (Index 0)
        print("\n--> TEST 1: Moving 'left_hip_pitch_joint' to +0.5")
        print("    LOOK FOR: Leg swinging Forward/Backward.")
        print("    (If it swings Sideways, your Index 0 is actually Roll)")

        start_time = time.time()
        while time.time() - start_time < 4.0:
            # 1. PIN THE BASE (Force position/orientation)
            data.qpos[0:3] = np.array([0.0, 0.0, 1.0])  # Hover at 1m height
            data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # Flat orientation
            data.qvel[0:6] = 0.0  # Stop base velocity

            # 2. Set Target
            target_pos = np.zeros(6)
            target_pos[0] = 0.5  # Move Index 0 (Hip Pitch)
            data.ctrl[:] = target_pos

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        # 2. Test HIP ROLL (Index 1)
        print("\n--> TEST 2: Moving 'left_hip_roll_joint' to +0.5")
        print("    LOOK FOR: Leg swinging Sideways (Out/In).")

        start_time = time.time()
        while time.time() - start_time < 4.0:
            # 1. PIN THE BASE
            data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
            data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            data.qvel[0:6] = 0.0

            # 2. Set Target
            target_pos = np.zeros(6)
            target_pos[1] = 0.5  # Move Index 1 (Hip Roll)
            data.ctrl[:] = target_pos

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

        # 3. Test KNEE (Index 2)
        print("\n--> TEST 3: Moving 'left_knee_joint' to +0.5")
        print("    LOOK FOR: Knee Bending.")
        print(
            "    (If it bends BACKWARDS like a bird, that's usually correct for +0.5)"
        )

        while viewer.is_running():
            # 1. PIN THE BASE
            data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
            data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
            data.qvel[0:6] = 0.0

            # 2. Set Target
            target_pos = np.zeros(6)
            target_pos[2] = 0.5  # Move Index 2 (Knee)
            data.ctrl[:] = target_pos

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
