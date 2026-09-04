"""
Custom robosuite environment: ReachTask
Robot must move its end-effector to touch a small target block on the table.
No camera, state-based observations only.

Usage:
    import robosuite as suite
    from src.simulation.reach_env import ReachTask

    env = suite.make("ReachTask", robots="Panda", has_renderer=True, ...)
"""
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.base import register_env


@register_env
class ReachTask(ManipulationEnv):
    """
    Move the end-effector to touch a small target block on the table.

    Observations (state-based, no camera):
        robot0_eef_pos       (3,)  end-effector position
        robot0_eef_quat      (4,)  end-effector orientation
        robot0_gripper_qpos  (2,)  gripper finger positions
        target_pos           (3,)  target block centre position
        eef_to_target        (3,)  vector from eef to target
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        reach_threshold=0.05,           # success if eef within 5 cm of target
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=300,
        ignore_done=True,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reach_threshold = reach_threshold
        self.reward_shaping = reward_shaping

        self.use_object_obs = use_object_obs


        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )


    # Build the scene

    def _load_model(self):
        super()._load_model()

        # Position robot at table edge
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Sample a random target XY position each episode
        tx = self.rng.uniform(-0.15, 0.15) + self.table_offset[0]
        ty = self.rng.uniform(-0.15, 0.15) + self.table_offset[1]
        tz = self.table_offset[2] + 0.1

        self._target_pos = np.array([tx, ty, tz])

        # Semi-transparent sphere - reach the EEF into this
        sphere_site = ET.SubElement(mujoco_arena.worldbody, "site")
        sphere_site.set("name", "reach_target")
        sphere_site.set("pos", f"{tx} {ty} {tz}")
        sphere_site.set("size", f"{self.reach_threshold}")
        sphere_site.set("type", "sphere")
        sphere_site.set("rgba", "0.2 0.9 0.2 0.4")

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )


    # References

    def _setup_references(self):
        super()._setup_references()


    # Reset

    def _reset_internal(self):
        super()._reset_internal()


    # Reward

    def reward(self, action=None):
        if self._check_success():
            return 1.0

        if self.reward_shaping:
            eef_pos = np.array(self.sim.data.site_xpos[
                self.robots[0].eef_site_id["right"]
                if isinstance(self.robots[0].eef_site_id, dict)
                else self.robots[0].eef_site_id
            ])
            dist = np.linalg.norm(eef_pos - self._target_pos)
            return -dist

        return 0.0


    # Success check
    
    def _check_success(self):
        eef_pos = np.array(self.sim.data.site_xpos[
            self.robots[0].eef_site_id["right"]
            if isinstance(self.robots[0].eef_site_id, dict)
            else self.robots[0].eef_site_id
        ])
        dist = np.linalg.norm(eef_pos - self._target_pos)
        return bool(dist < self.reach_threshold)


    # Observations
    
    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def target_pos(obs_cache):
                return self._target_pos.copy()

            @sensor(modality=modality)
            def eef_to_target(obs_cache):
                eef = np.array(self.sim.data.site_xpos[
                    self.robots[0].eef_site_id["right"]
                    if isinstance(self.robots[0].eef_site_id, dict)
                    else self.robots[0].eef_site_id
                ])
                return self._target_pos - eef

            for s in [target_pos, eef_to_target]:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables