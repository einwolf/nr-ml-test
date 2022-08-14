import os
import random

import gym
import numpy as np
from gym import Env
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (EvalCallback,
                                                StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecTransposeImage)

log_path = "tensorboard_logs"
ppo_model_path = os.path.join("saved_models", "ppo_model_shower")

def make_output_dirs():
    os.makedirs(log_path, exist_ok=True)
    # os.makedirs(ppo_model_path, exist_ok=True)


class ShowerEnv(Env):
    """
    Shower temperature environment
    """
    def __init__(self):
        # Actions we can take: down, stay, up
        self.action_space = Discrete(3)

        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))

        # Set start temp
        self.state = 38 + random.randint(-3,3)

        # Set shower length
        self.shower_length = 60
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        self.state += action -1 
        # Reduce shower length by 1 second
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self, new=True):
        # Implement visualization
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([38 + random.randint(-3,3)]).astype(float)
        # Reset shower time
        self.shower_length = 60 
        return self.state


def env_test():
    """
    Test environment with random actions
    """
    # Space examples
    Discrete(3)
    d1 = Box(0,1,shape=(3,3)).sample()
    d2 = Box(0,255,shape=(3,3), dtype=int).sample()
    d3 = Tuple((Discrete(2), Box(0,100, shape=(1,)))).sample()
    d4 = Dict({'height':Discrete(2), "speed":Box(0,100, shape=(1,))}).sample()
    d5 = MultiBinary(4).sample()
    d6 = MultiDiscrete([5,2,2]).sample()

    env = ShowerEnv()

    # Custom environment checker
    # check_env(env, warn=True)

    # Test environment
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

    env = Monitor(ShowerEnv())
    env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(env, 
                                callback_on_new_best=stop_callback, 
                                eval_freq=1000, 
                                best_model_save_path=ppo_model_path, 
                                verbose=1)

    # The training and eval env mismatch is normal
    model.learn(total_timesteps=40000, callback=eval_callback)

    # print(f"Save {ppo_model_path}")
    # model.save(ppo_model_path)


def eval():
    """
    Evaluate model training
    """
    make_output_dirs()

    env = Monitor(ShowerEnv())
    env = DummyVecEnv([lambda: env])

    print(f"Load best model {ppo_model_path}")
    model = PPO.load(f"{ppo_model_path}/best_model.zip", env=env)

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
