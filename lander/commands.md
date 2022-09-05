# Command buffer

## Monitor

```bash
nvidia-smi dmon -d 4
```

## Check env

```bash
lander-env_test
```

## Train

```bash
lander-train --total_timesteps 40000 --save_model saved_models/dqn/last_train.zip
lander-train --total_timesteps 1000000 --reward_threshold 1000 --save_model saved_models/dqn/last_train.zip
lander-train --total_timesteps 1000000 --reward_threshold 1000 --load_model saved_models/dqn/best_model.zip --save_model saved_models/dqn/last_train.zip
```

## Evaluation

```bash
lander-eval --iterations 2 --load_model saved_models/dqn/best_model.zip
```
