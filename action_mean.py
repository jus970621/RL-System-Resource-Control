#!/usr/bin/env python3

import env as env_module
import numpy as np
import pandas as pd
import torch

data_path = "output/expert/2026-04-03/expert_merged.csv"
data_frame = pd.read_csv(data_path)

rl_env = env_module.Env(data_frame)
state = rl_env.reset()

action_rewards = [[], [], []]
#print(len(data_frame))
episode_lengths = 100


for action in [0, 1, 2]:
    rl_env = env_module.Env(data_frame)
    state = rl_env.reset()
    done = False

    rewards = []

    while not done:
        next_state, reward, done = rl_env.step(action)
        rewards.append(reward)
        state = next_state

    print(f"Action {action} avg reward:", np.mean(rewards), "steps:", len(rewards))
