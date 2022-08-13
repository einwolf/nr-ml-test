import os

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

environment_name = "Breakout-v0"

log_path = "tensorboard_logs"
a2c_model_path = os.path.join("saved_models", "a2c_model_breakout")

def make_output_dirs():
    os.makedirs(log_path, exist_ok=True)
    # os.makedirs(a2c_model_path, exist_ok=True)

def env_test():
    """
    Test environment with random actions
    """
    env = gym.make(environment_name)

    episodes = 10
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0 
        
        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score+=reward
        print('Episode:{} Score:{}'.format(episode, score))

    env.close()

    env.action_space.sample()

    env.observation_space.sample()


def train():
    """
    Train model
    """
    make_output_dirs()

    env = make_atari_env(environment_name, n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_path)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=20, verbose=1)
    eval_callback = EvalCallback(env, 
                                callback_on_new_best=stop_callback, 
                                eval_freq=1000, 
                                best_model_save_path=a2c_model_path, 
                                verbose=1)

    # The training and eval env mismatch is normal
    model.learn(total_timesteps=40000, callback=eval_callback)

    # print(f"Save {a2c_model_path}")
    # model.save(a2c_model_path)


def eval():
    """
    Evaluate model training
    """
    make_output_dirs()

    env = make_atari_env(environment_name, n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)

    print(f"Load best model {a2c_model_path}")
    model = A2C.load("saved_models/a2c_model_breakout/best_model.zip", env=env)

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
