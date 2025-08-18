from envs.robot_arm_env import RobotArmEnv

env = RobotArmEnv("lowCostRobotArm/robotScene.xml")

obs, info = env.reset()
done = False

for _ in range(1000):
    action = env.action_space.sample()   # random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
