import threading
import time

import numpy as np

from sic_framework.devices.franka import Franka
from sic_framework.devices.common_franka.franka_motion import (
    FrankaPose,
    FrankaPoseRequest,
)


class FrankaReal:

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

        self.lock = threading.Lock()

        self.franka.motion.register_callback(
            self._on_pose
        )

        self.franka.motion.request(
            FrankaPoseRequest(stream=True)
        )

        while self.position is None:
            time.sleep(0.1)

    def _on_pose(self, pose):

        with self.lock:

            self.position = np.asarray(
                pose.position,
                dtype=np.float32,
            )

            self.orientation = np.asarray(
                pose.orientation,
                dtype=np.float32,
            )

    def get_ee_position(self):

        with self.lock:
            return self.position.copy()

    def apply_action(self, action):

        with self.lock:

            position = self.position.copy()
            orientation = self.orientation.copy()

        delta = 0.01 * np.asarray(
            action,
            dtype=np.float32,
        )

        target = position + delta

        target = np.clip(
            target,
            self.WORKSPACE_MIN,
            self.WORKSPACE_MAX,
        )

        self.franka.motion.send_message(
            FrankaPose(
                position=target,
                orientation=orientation,
            )
        )

    def reset(self):

        with self.lock:
            orientation = self.orientation.copy()

        self.franka.motion.send_message(
            FrankaPose(
                position=self.HOME_POSITION,
                orientation=orientation,
            )
        )

        time.sleep(2.0)