import os
import torch
import numpy as np
from gymnasium.wrappers import TimeLimit  # Updated: use gymnasium instead of gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.robot_arm_env import RobotArmEnv  # your env file

LOG_DIR = "logs/ppo_robot_arm"
MODEL_DIR = "models/ppo_robot_arm"
EVAL_DIR = "eval/ppo_robot_arm"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

SEED = 42
TIMESTEPS = 200_000  # bump as needed
EP_LEN = 200         # max steps per episode

def make_env(render_mode=None):
    def _init():
        env = RobotArmEnv(xml_path=r"C:\Users\Cyril\PycharmProjects\RobotArm\lowCostRobotArm\robotScene.xml")
        # TimeLimit to define episodes - now using gymnasium.wrappers.TimeLimit
        env = TimeLimit(env, max_episode_steps=EP_LEN)
        # Monitor to record ep_rew_mean, ep_len, etc.
        env = Monitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"), allow_early_resets=True)
        return env
    return _init

if __name__ == "__main__":
    # Vectorize (even if 1 env) so SB3 logging is smooth and extensible
    env = DummyVecEnv([make_env()])
    eval_env = DummyVecEnv([make_env()])

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=None,  # Disable tensorboard logging to avoid tensorboard dependency issues
        seed=SEED,
        learning_rate=3e-4,
        n_steps=2048,          # collect per update
        batch_size=64,
        n_epochs=10,           # minibatch passes per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        device="auto",
    )

    # Save periodic checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,               # every N steps
        save_path=MODEL_DIR,
        name_prefix="ppo_robot_arm",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Evaluate every X steps and save the best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=EVAL_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=TIMESTEPS, callback=[checkpoint_cb, eval_cb])

    # Final save
    model.save(os.path.join(MODEL_DIR, "ppo_robot_arm_final"))
    print("Training complete. Model saved.")
