"""
Custom robosuite environment: PushTask
Robot must push a box to a goal region on the table.
No camera

Usage:
    import robosuite as suite
    from src.simulation.push_env import PushTask
    import robosuite.environments  # triggers registration

    # Register and create
    env = suite.make("PushTask", robots="Panda", has_renderer=True, ...)
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
class PushTask(ManipulationEnv):
    """
    Push a box from its spawn position to a fixed goal region.

    Observations (state-based, no camera):
        robot0_eef_pos (3,) end-effector position
        robot0_eef_quat (4,) end-effector orientation
        robot0_gripper_qpos (2,) gripper finger positions
        box_pos (3,) box centre position
        box_to_goal (3,) vector from box to goal
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
        goal_pos=(0.2, 0.15, 0.0),     # XY offset from table centre; Z ignored
        goal_threshold=0.05,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=500,
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
        self.table_full_size   = table_full_size
        self.table_friction    = table_friction
        self.table_offset      = np.array((0, 0, 0.8))
        self.goal_offset       = np.array(goal_pos[:2])   # XY only
        self.goal_threshold    = goal_threshold
        self.reward_shaping    = reward_shaping
        self.use_object_obs    = use_object_obs
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


        # Visual goal marker
        goal_x = self.table_offset[0] + self.goal_offset[0]
        goal_y = self.table_offset[1] + self.goal_offset[1]
        goal_z = self.table_offset[2]
        goal_site = ET.SubElement(mujoco_arena.worldbody, "site")
        goal_site.set("name", "goal_marker")
        goal_site.set("pos", f"{goal_x} {goal_y} {goal_z}")
        goal_site.set("size", f"{self.goal_threshold} {self.goal_threshold} 0.001")
        goal_site.set("type", "cylinder")
        goal_site.set("rgba", "0.2 0.9 0.2 0.6")

        self.box = BoxObject(
            name="box",
            size_min=(0.025, 0.025, 0.025),
            size_max=(0.025, 0.025, 0.025),
            rgba=[0.8, 0.2, 0.2, 1],
            rng=self.rng,
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.box)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.box,
                x_range=[-0.05, 0.05],
                y_range=[-0.05, 0.05],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.box,
        )


    # References
    
    def _setup_references(self):
        super()._setup_references()
        self.box_body_id = self.sim.model.body_name2id(self.box.root_body)
        # Goal position = table centre + XY offset, at table surface height
        self.goal_pos_world = np.array([
            self.table_offset[0] + self.goal_offset[0],
            self.table_offset[1] + self.goal_offset[1],
            self.table_offset[2],
        ])


    # Reset
    
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                )


    # Reward
    
    def reward(self, action=None):
        box_pos  = np.array(self.sim.data.body_xpos[self.box_body_id])
        dist_box_to_goal = np.linalg.norm(box_pos[:2] - self.goal_pos_world[:2])

        if self._check_success():
            return 1.0

        if self.reward_shaping:
            r_dist = -dist_box_to_goal

            # Use robosuite helper to get gripper-to-box distance
            eef_to_box = self._gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.box.root_body,
                target_type="body",
                return_distance=True,
            )
            r_contact = -(eef_to_box * 0.3)
            return r_dist + r_contact

        return 0.0


    # Success check
    
    def _check_success(self):
        box_pos  = np.array(self.sim.data.body_xpos[self.box_body_id])
        dist_xy  = np.linalg.norm(box_pos[:2] - self.goal_pos_world[:2])
        return bool(dist_xy < self.goal_threshold)


    # Observations
    
    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def box_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.box_body_id])

            @sensor(modality=modality)
            def box_to_goal(obs_cache):
                b = np.array(self.sim.data.body_xpos[self.box_body_id])
                return self.goal_pos_world - b

            for s in [box_pos, box_to_goal]:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables
