import argparse
import os

import gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback,
    StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecTransposeImage)

environment_name = "LunarLander-v2"

log_path = "tensorboard_logs"
dqn_model_path = os.path.join("saved_models", "dqn")

def make_output_dirs():
    os.makedirs(log_path, exist_ok=True)
    # os.makedirs(a2c_model_path, exist_ok=True)


def main():
    """
    Train model
    """
    # Parse command line
    args = parse_cmd_line()

    if args.reward_threshold == 0:
        args.reward_threshold = False

    print(f"{args.total_timesteps=}")
    print(f"{args.reward_threshold=}")

    # Initialize
    make_output_dirs()

    env = gym.make(environment_name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

    # Most callbacks need to attach to a EvalCallback and will complain if not.
    # This means one callback each on callback_after_eval / callback_on_new_best.
    # The CallbackList only works as a parameter to learn(callback=).
    checkpoint_eval_freq = 1000
    checkpoint_callback = CheckpointCallback(save_freq=1, name_prefix="checkpoint", save_path=dqn_model_path, verbose=1)
    checkpoint_eval = EvalCallback(env,
                            eval_freq=checkpoint_eval_freq,
                            callback_after_eval=checkpoint_callback,
                            verbose=1)

    stop_no_improvement = StopTrainingOnNoModelImprovement(5, min_evals=10, verbose=1)
    stop_no_improvement_eval = EvalCallback(env,
                            eval_freq=checkpoint_eval_freq,
                            callback_on_new_best=stop_no_improvement,
                            best_model_save_path=dqn_model_path,
                            verbose=1)

    stop_reward_callback = StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold, verbose=1)
    stop_reward_eval = EvalCallback(env,
                            eval_freq=checkpoint_eval_freq,
                            callback_on_new_best=stop_reward_callback,
                            best_model_save_path=dqn_model_path,
                            verbose=1)

    callbacks = []
    callbacks.append(checkpoint_eval)
    callbacks.append(stop_no_improvement_eval)
    if args.reward_threshold:
        callbacks.append(stop_reward_eval)
    
    callback_list = CallbackList(callbacks)

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callback_list)

    # print(f"Save {a2c_model_path}")
    # model.save(a2c_model_path)


def parse_cmd_line():
    parser = argparse.ArgumentParser(description="Training phase")

    parser.add_argument("--total_timesteps", required=True, type=int,
                    help="Maximum number of training time steps")
    parser.add_argument("--reward_threshold", required=False, type=int, default=0,
                    help="Stop when reward reaches this value")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
    