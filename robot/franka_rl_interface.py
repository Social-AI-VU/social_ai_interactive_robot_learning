import threading
import time

import numpy as np

from sic_framework.devices.franka import Franka
from sic_framework.devices.common_franka.franka_motion import (
    FrankaPose,
    FrankaPoseRequest,
)


class FrankaRLInterface:

    WORKSPACE_MIN = np.array(
        [0.30, -0.40, 0.10],
        dtype=np.float32,
    )

    WORKSPACE_MAX = np.array(
        [0.80, 0.40, 0.70],
        dtype=np.float32,
    )

    HOME_POSITION = np.array(
        [0.50, 0.00, 0.40],
        dtype=np.float32,
    )

    def __init__(self):

        self.franka = Franka()

        self.position = None
        self.orientation = None

        self.pose_lock = threading.Lock()

        self.franka.motion.register_callback(
            self._on_pose
        )

        self.franka.motion.request(
            FrankaPoseRequest(stream=True)
        )

        print("Waiting for Franka pose stream...")

        while self.position is None:
            time.sleep(0.1)

        print("Franka connected.")

    def _on_pose(self, pose):

        with self.pose_lock:

            self.position = np.array(
                pose.position,
                dtype=np.float32,
            )

            self.orientation = np.array(
                pose.orientation,
                dtype=np.float32,
            )

    def get_ee_position(self):

        with self.pose_lock:
            return self.position.copy()

    def apply_action(self, action):

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        with self.pose_lock:

            current_position = (
                self.position.copy()
            )

            current_orientation = (
                self.orientation.copy()
            )

        # 1 cm max motion per RL step
        delta = 0.01 * action

        new_position = (
            current_position + delta
        )

        new_position = np.clip(
            new_position,
            self.WORKSPACE_MIN,
            self.WORKSPACE_MAX,
        )

        self.franka.motion.send_message(
            FrankaPose(
                position=new_position,
                orientation=current_orientation,
            )
        )

    def reset(self):

        with self.pose_lock:
            orientation = (
                self.orientation.copy()
            )

        self.franka.motion.send_message(
            FrankaPose(
                position=self.HOME_POSITION,
                orientation=orientation,
            )
        )

        time.sleep(2.0)