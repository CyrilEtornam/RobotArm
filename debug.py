from envs.robot_arm_env import RobotArmEnv

env = RobotArmEnv("low_cost_robot_arm/robot_scene.xml")

obs, info = env.reset()
done = False

for _ in range(1000):
    action = env.action_space.sample()   # random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
