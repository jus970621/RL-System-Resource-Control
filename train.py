#!/usr/bin/env python3
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import env as env_module

data_path = "output/expert/2026-04-03/expert_merged.csv"
data_frame = pd.read_csv(data_path)
device = torch.device("cpu")

rl_env = env_module.Env(data_frame)
state = rl_env.reset()

state_dim = len(rl_env.state_cols)
action_dim = len(rl_env.actions)

memory = deque(maxlen=10000)

num_episodes = 100
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
target_update = 10

print("state_dim:", state_dim)
print("action_dim:", action_dim)
print("memory:", memory, type(memory))

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def select_action(model, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())

def train_step(model, target_model, optimizer, memory, batch_size, gamma):
    if len(memory) < batch_size:
        return None

    batch = random.sample(memory, batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = model(states).gather(1, actions)

    with torch.no_grad():
        next_q_values = target_model(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.SmoothL1Loss()(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 👈 여기 추가
    optimizer.step()

    return loss.item()

def evaluate_policy(model, env, num_episodes=5):
    model.eval()
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        action_counts = [0, 0, 0]

        while not done:
            action = select_action(model, state, epsilon=0.0, action_dim=action_dim)
            action_counts[action] += 1

            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        print(f"[EVAL] Episode {episode:03d} | Reward: {total_reward:.2f} | Actions: {action_counts}")

    model.train()
    print(f"[EVAL] Mean Reward: {np.mean(rewards):.2f}")
    return rewards

def save_model(model):
    torch.save(model.state_dict(), "dqn_model.pth")
    print("Model saved: dqn_model.pth")


model = DQN(state_dim, action_dim).to(device)
target_model = DQN(state_dim, action_dim).to(device)
target_model.load_state_dict(model.state_dict())

model.train()
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
last_loss = None

episode_rewards = []

for episode in range(num_episodes):
    state = rl_env.reset()
    done = False
    total_reward = 0.0
    last_loss = None
    action_counts = [0, 0, 0]

    while not done:
        action = select_action(model, state, epsilon, action_dim)
        action_counts[action] += 1
        next_state, reward, done = rl_env.step(action)

        memory.append((state, action, reward, next_state, done))

        loss = train_step(model, target_model, optimizer, memory, batch_size, gamma)

        if loss is not None:
            last_loss = loss

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    avg_reward_10 = np.mean(episode_rewards[-10:])

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % target_update == 0:
        target_model.load_state_dict(model.state_dict())

    print(
        f"Episode {episode:03d} | Reward: {total_reward:.2f} "
        f"| Avg10: {avg_reward_10:.2f} "
        f"| Actions: {action_counts} "
        f"| Epsilon: {epsilon:.3f} | Loss: {last_loss}"
    )

print("\n=== Training Finished ===")
print(f"Final Avg10 Reward: {np.mean(episode_rewards[-10:]):.2f}")

save_model(model)

eval_rewards = evaluate_policy(model, rl_env, num_episodes=5)
