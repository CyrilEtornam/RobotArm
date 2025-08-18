from envs.robot_arm_env import RobotArmEnv

env = RobotArmEnv()
obs, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # random
    obs, reward, done, truncated, info = env.step(action)
    env.render()
