import mujoco
import mujoco.viewer
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
robot_model_path = "mujoco_xml/SF_biped.xml"


def get_projected_gravity(quaternion):
    # Check for zero quaternion (invalid data)
    if np.linalg.norm(quaternion) < 1e-6:
        return np.array([0.0, 0.0, 0.0])

    # quaternion = [x, y, z, w]
    r = R.from_quat(quaternion)
    g_world = np.array([0.0, 0.0, -1.0])
    g_base = r.apply(g_world, inverse=True)
    return g_base.flatten()


def main():
    print(f"Loading Model: {robot_model_path}")
    model = mujoco.MjModel.from_xml_path(robot_model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 1. Reset
        mujoco.mj_resetData(model, data)

        # 2. Set Upright Pose
        data.qpos[2] = 1.0  # Lift up (z)
        # Set quaternion to Identity [w, x, y, z] = [1, 0, 0, 0]
        data.qpos[3] = 1.0
        data.qpos[4] = 0.0
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0

        # --- CRITICAL FIX: Update Sensors ---
        # This propagates the new qpos to the sensors immediately
        mujoco.mj_forward(model, data)
        # ------------------------------------

        print("------------------------------------------------")
        print("GRAVITY VECTOR TEST")
        print("Keep the robot UPRIGHT.")
        print("Expected: [0.0, 0.0, -1.0]")
        print("------------------------------------------------")

        while viewer.is_running():
            # Read Orientation
            # Note: MuJoCo usually returns [w, x, y, z] or [x, y, z, w] depending on sensor config
            # But standard 'framequat' or 'orientation' sensor is usually [w, x, y, z]
            quat_raw = data.sensor("orientation").data.copy()

            # Scipy expects [x, y, z, w]
            # We map [w, x, y, z] -> [x, y, z, w]
            quat = np.array([quat_raw[1], quat_raw[2], quat_raw[3], quat_raw[0]])

            # Calculate Gravity
            gravity = get_projected_gravity(quat)

            # Print cleanly
            print(f"Gravity: [{gravity[0]:.2f}, {gravity[1]:.2f}, {gravity[2]:.2f}]")

            # Physics step
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.1)


if __name__ == "__main__":
    main()
