#!/usr/bin/env python3
import env as env_module
import pandas as pd
import numpy as np

data_path = "output/expert/2026-04-03/expert_merged.csv"
df = pd.read_csv(data_path)

rl_env = env_module.Env(df)

state = rl_env.reset()
done = False

total_reward = 0
action_counts = [0, 0, 0]

while not done:
    row = rl_env.df.iloc[rl_env.idx]

    cpu_idle = row["cpu_idle"]
    mem_pressure = row["mem_pressure"]

    # 🔥 rule policy
    if cpu_idle > 90 and mem_pressure < 0.65:
        action = 2  # increase
    elif cpu_idle < 75 or mem_pressure > 0.75:
        action = 1  # decrease
    else:
        action = 0  # hold

    action_counts[action] += 1

    next_state, reward, done = rl_env.step(action)
    total_reward += reward

    state = next_state

print("Total Reward:", total_reward)
print("Action Counts:", action_counts)