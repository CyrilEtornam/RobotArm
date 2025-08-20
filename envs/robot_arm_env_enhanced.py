import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import spaces


class RobotArmEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, xml_path=r"C:\Users\Cyril\PycharmProjects\RobotArm\lowCostRobotArm\robotScene.xml"):
        super(RobotArmEnv, self).__init__()

        # Load model
        fullpath = os.path.abspath(xml_path)
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)

        # Viewer setup (lazy init)
        self.viewer = None

        # ----- ACTION SPACE -----
        # actions = 6 joint controls + 1 gripper control in range [-1, 1]
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
        
        # Set initial gripper to open position
        gripper_idx = self._get_gripper_actuator_idx()
        if gripper_idx is not None:
            self.data.ctrl[gripper_idx] = -1.0  # Open gripper
        
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

        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
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

    def _get_gripper_actuator_idx(self):
        """Get the index of the gripper actuator."""
        for i in range(self.model.nu):  # Iterate over the number of actuators
            name = self.model.actuator_acc0[i]  # Use actuator_acc0 or another appropriate attribute
            if 'gripper' in str(name).lower():
                return i
        return None

    def _compute_reward(self):
        gripper_pos = self.data.site("gripper_tip").xpos
        cube_pos = self.data.body("cube").xpos
        goal_pos = self.data.site("goal_site").xpos

        dist_gripper_cube = np.linalg.norm(gripper_pos - cube_pos)
        dist_cube_goal = np.linalg.norm(cube_pos - goal_pos)

        # Base reward for gripper positioning
        reward = -1.0 * dist_gripper_cube

        # Enhanced grasping rewards
        if self.is_cube_in_grasp_range():
            # Strong reward for being in grasp position
            reward += 5.0
            
            # Additional reward for gripper closing when in position
            gripper_idx = self._get_gripper_actuator_idx()
            if gripper_idx is not None:
                gripper_action = self.data.ctrl[gripper_idx]
                gripper_range = self.model.actuator_ctrlrange[gripper_idx]
                normalized_gripper = (gripper_action - gripper_range[0]) / (gripper_range[1] - gripper_range[0])
                
                # Reward for closing gripper (positive values = closing)
                if normalized_gripper > 0.5:
                    reward += 10.0 * normalized_gripper

        # Grasp success rewards
        if self.is_cube_grasped():
            reward += 20.0
            # Additional reward for maintaining grasp
            reward -= dist_cube_goal * 2.0

        # Goal achievement rewards
        if dist_cube_goal < 0.1:
            reward += 15.0
        if dist_cube_goal < 0.05:
            reward += 30.0

        return reward

    def is_cube_in_grasp_range(self, position_threshold=0.025, height_threshold=0.015):
        """Check if cube is in optimal grasping position."""
        gripper_pos = self.data.site("gripper_tip").xpos
        cube_pos = self.data.body("cube").xpos

        # Check horizontal distance
        horizontal_dist = np.linalg.norm(gripper_pos[:2] - cube_pos[:2])
        
        # Check height alignment
        height_diff = gripper_pos[2] - cube_pos[2]
        
        return horizontal_dist < position_threshold and abs(height_diff) < height_threshold

    def is_cube_grasped(self, threshold=0.02):
        """Enhanced grasp detection considering gripper state."""
        gripper_pos = self.data.site("gripper_tip").xpos
        cube_pos = self.data.body("cube").xpos
        
        # Distance check
        dist = np.linalg.norm(gripper_pos - cube_pos)
        
        # Height check
        height_diff = abs(gripper_pos[2] - cube_pos[2])
        
        # Gripper state check
        gripper_idx = self._get_gripper_actuator_idx()
        if gripper_idx is not None:
            gripper_action = self.data.ctrl[gripper_idx]
            gripper_range = self.model.actuator_ctrlrange[gripper_idx]
            normalized_gripper = (gripper_action - gripper_range[0]) / (gripper_range[1] - gripper_range[0])
            
            # Consider grasped if close and gripper is closing
            return dist < threshold and height_diff < 0.015 and normalized_gripper > 0.3
        
        return dist < threshold and height_diff < 0.015
