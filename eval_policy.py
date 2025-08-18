# eval_policy.py
import os
import time
import numpy as np
from stable_baselines3 import PPO
from envs.robot_arm_env import RobotArmEnv

MODEL_DIR = "models/ppo_robot_arm"
MODEL_NAME = "best_model"  # or "ppo_robot_arm_final"

def run_episode(env, model, render=True):
    obs, info = env.reset()
    ep_rew = 0.0
    for t in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_rew += reward
        if render:
            env.render()
            # throttle a bit so the viewer can keep up
            time.sleep(0.01)
        if terminated or truncated:
            break
    return ep_rew

if __name__ == "__main__":
    env = RobotArmEnv("scene.xml")
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    model = PPO.load(model_path, env=None, device="auto")  # env not required for inference

    # run a few episodes
    scores = []
    for _ in range(3):
        score = run_episode(env, model, render=True)
        print(f"Episode reward: {score:.2f}")
        scores.append(score)
    print(f"Avg reward: {np.mean(scores):.2f}")
