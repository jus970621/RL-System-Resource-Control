#!/usr/bin/env python3

import env as env_module
import numpy as np
import pandas as pd
import time

expert_example_path = "output/expert/2026-04-03/expert_merged.csv"
expert_example = pd.read_csv(expert_example_path)

rl_env = env_module.Env(expert_example)

reward_box = []

print("state_dim:", len(rl_env.state_cols))

for episode in range(100):
    state = rl_env.reset()
    done = False
    total_reward = 0.0
    action_counts = [0, 0, 0]
    step_count = 0

    while not done:
        action = np.random.randint(0, 3)
        action_counts[action] += 1

        next_state, reward, done = rl_env.step(action)

        total_reward += reward
        state = next_state
        step_count += 1

    reward_box.append(total_reward)
    avg_reward = np.mean(reward_box)
    avg_reward_10 = np.mean(reward_box[-10:])

    print(
        f"Random Episode {episode:03d} | Reward: {total_reward:.2f} "
        f"| Avg: {avg_reward:.2f} | Avg10: {avg_reward_10:.2f} "
        f"| Actions: {action_counts} | Steps: {step_count}"
    )

print("Random rewards:", reward_box)
print("Random mean reward:", np.mean(reward_box))