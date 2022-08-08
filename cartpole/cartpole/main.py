import os
import gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import (EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# CartPole action and observation space
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
#
# Action space
# | Num | Action                 |
# |-----|------------------------|
# | 0   | Push cart to the left  |
# | 1   | Push cart to the right |
#
# Observation space
# | Num | Observation           | Min                 | Max               |
# |-----|-----------------------|---------------------|-------------------|
# | 0   | Cart Position         | -4.8                | 4.8               |
# | 1   | Cart Velocity         | -Inf                | Inf               |
# | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
# | 3   | Pole Angular Velocity | -Inf                | Inf               |

log_path = "tensorboard_logs"
ppo_model_path = os.path.join("saved_models", "ppo_model_cartpole")

def make_output_dirs():
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)


def train():
    """
    Train model
    """
    make_output_dirs()
    environment_name = 'CartPole-v0'

    env = gym.make(environment_name)
    env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

    model.learn(total_timesteps=20000)

    print("Save PPO model")
    model.save(ppo_model_path)


def eval():
    """
    Evaluate model training
    """
    make_output_dirs()
    environment_name = 'CartPole-v0'

    env = gym.make(environment_name)

    print(f"Load model {ppo_model_path}")
    model = PPO.load(ppo_model_path, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"mean_reward per episode = {mean_reward}")
    print(f"std_reward per episode = {std_reward}")

    env.close()


def train2():
    """
    Train model with PPO and automatic stopping callback
    """
    make_output_dirs()
    environment_name = 'CartPole-v0'

    env = gym.make(environment_name)
    env = DummyVecEnv([lambda: env])
    model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)
    eval_callback = EvalCallback(env, 
                                callback_on_new_best=stop_callback, 
                                eval_freq=10000, 
                                best_model_save_path=ppo_model_path, 
                                verbose=1)


    model.learn(total_timesteps=20000, callback=eval_callback)
    del model

    print("Load best model")
    best_model_path = os.path.join(ppo_model_path, "best_model")
    model = PPO.load(best_model_path, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"mean_reward per episode = {mean_reward}")
    print(f"std_reward per episode = {std_reward}")

    env.close()


def train3():
    """
    Train model with DQN and automatic stopping callback
    """
    make_output_dirs()
    environment_name = 'CartPole-v0'

    env = gym.make(environment_name)
    env = DummyVecEnv([lambda: env])
    model = DQN('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)
    eval_callback = EvalCallback(env, 
                                callback_on_new_best=stop_callback, 
                                eval_freq=10000, 
                                best_model_save_path=ppo_model_path, 
                                verbose=1)


    model.learn(total_timesteps=200000, callback=eval_callback)
    del model

    print("Load best model")
    best_model_path = os.path.join(ppo_model_path, "best_model")
    model = DQN.load(best_model_path, env=env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print(f"mean_reward per episode = {mean_reward}")
    print(f"std_reward per episode = {std_reward}")

    env.close()
