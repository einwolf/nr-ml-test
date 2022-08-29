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


if __name__ == "__main__":
    main()
