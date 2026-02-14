import numpy as np
from collections import deque
import torch
import mujoco
from scipy.spatial.transform import Rotation as R
from typing import List


class HistoryBuffer:
    def __init__(self, history_len: int):
        self.history_len = history_len
        self.buffer = deque(maxlen=history_len)

    def update_history(self, current_frame: np.ndarray):
        self.buffer.appendleft(current_frame)

    def reset(self):
        self.buffer.clear()

    def get_stacked_obs(self) -> np.ndarray:
        # fill the buffer with copies of the first frame if it's not full
        while len(self.buffer) < self.history_len:
            self.buffer.append(self.buffer[0])

        stacked = np.concatenate(list(self.buffer))
        return stacked.astype(np.float32)


def get_mujoco_data(
    model: mujoco.MjModel, data: mujoco.MjData, joint_names: List[str]
) -> np.ndarray:
    quat_raw = data.sensor("orientation").data.copy()
    quat = np.array(
        [quat_raw[1], quat_raw[2], quat_raw[3], quat_raw[0]], dtype=np.float32
    )

    # angular velocity
    ang_vel = data.sensor("angular-velocity").data.copy()

    # get joint positions and joint velocities
    joints_pos = []
    joints_vel = []

    for name in joint_names:
        addr_pos = model.joint(name).qposadr
        addr_vel = model.joint(name).dofadr
        joints_pos.append(data.qpos[addr_pos])
        joints_vel.append(data.qvel[addr_vel])

    joints_pos = np.array(joints_pos)
    joints_vel = np.array(joints_vel)

    return (
        quat.flatten(),
        ang_vel.flatten(),
        joints_pos.flatten(),
        joints_vel.flatten(),
    )


def get_projected_gravity(quat: np.ndarray) -> np.ndarray:
    r = R.from_quat(quat)
    g_world = np.array([0.0, 0.0, -1])
    g_base = r.apply(g_world, inverse=True)
    return g_base.flatten()
