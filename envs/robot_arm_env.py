import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces


class RobotArmEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, xml_path="low_cost_robot_arm/robot_scene.xml"):
        super(RobotArmEnv, self).__init__()

        # Load model
        fullpath = os.path.abspath(xml_path)
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)

        # Viewer setup (lazy init)
        self.viewer = None

        # ----- ACTION SPACE -----
        # actions = 6 joint controls in range [-1, 1]
        n_actuators = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actuators,), dtype=np.float32
        )

        # ----- OBSERVATION SPACE -----
        # obs = gripper_pos(3) + cube_pos(3) + goal_pos(3) + joint_pos(6) + joint_vel(6)
        obs_dim = 3 + 3 + 3 + self.model.nq + self.model.nv
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # ====================================================
    # Core API
    # ====================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Scale action to actuator range
        ctrl_range = self.model.actuator_ctrlrange
        lb, ub = ctrl_range[:, 0], ctrl_range[:, 1]
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)

        self.data.ctrl[:] = scaled_action
        mujoco.mj_step(self.model, self.data)

        reward = self._compute_reward()
        obs = self._get_obs()

        terminated = False   # No hard terminal state yet
        truncated = False    # Could add time-limit truncation

        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # just keep viewer open
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ====================================================
    # Helper functions
    # ====================================================
    def _get_obs(self):
        gripper_pos = self.data.site("gripper_tip").xpos.copy()
        cube_pos = self.data.body("cube").xpos.copy()
        goal_pos = self.data.site("goal_site").xpos.copy()

        joint_pos = self.data.qpos.copy()
        joint_vel = self.data.qvel.copy()

        return np.concatenate([gripper_pos, cube_pos, goal_pos, joint_pos, joint_vel])

    def _compute_reward(self):
        gripper_pos = self.data.site("gripper_tip").xpos
        cube_pos = self.data.body("cube").xpos
        goal_pos = self.data.site("goal_site").xpos

        dist_gripper_cube = np.linalg.norm(gripper_pos - cube_pos)
        dist_cube_goal = np.linalg.norm(cube_pos - goal_pos)

        reward = -dist_gripper_cube - dist_cube_goal

        if self.is_cube_grasped():
            reward += 2.0
        if dist_cube_goal < 0.05:
            reward += 5.0

        return reward

    def is_cube_grasped(self, threshold=0.03):
        """Check if cube is between gripper fingers (simple distance check)."""
        gripper_pos = self.data.site("gripper_tip").xpos
        cube_pos = self.data.body("cube").xpos
        return np.linalg.norm(gripper_pos - cube_pos) < threshold
