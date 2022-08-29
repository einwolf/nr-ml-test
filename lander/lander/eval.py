import os

import gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import (EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecTransposeImage)

environment_name = "LunarLander-v2"

log_path = "tensorboard_logs"
a2c_model_path = os.path.join("saved_models", "dqn_model_breakout")

def make_output_dirs():
    os.makedirs(log_path, exist_ok=True)
    # os.makedirs(a2c_model_path, exist_ok=True)


def main():
    """
    Evaluate model training
    """
    make_output_dirs()

    env = make_atari_env(environment_name, n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)

    print(f"Load best model {a2c_model_path}")
    model = A2C.load(f"{a2c_model_path}/best_model.zip", env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
    print(f"mean_reward per episode = {mean_reward}")
    print(f"std_reward per episode = {std_reward}")

    # Play game
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

    env.close()


if __name__ == "__main__":
    main()
    