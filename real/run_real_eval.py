#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
import torch.nn as nn

from env_real import RealEnv
import control


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "dqn_model_tuned.pth"
print("[INFO] model path:", MODEL_PATH)

device = torch.device("cpu")


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


def select_action(model, state):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)

    action = int(torch.argmax(q_values, dim=1).item())

    return action, q_values.squeeze(0).tolist()


def main():
    #step_interval >> 각 액션별 기다리는 시간 ex) 액션 hold 후 5초 대기 액션 increase 후 5초 대기 액션 decrease
    #max_steps = 한 에피소드(episode는 한번이니 의미 없음)당 학습하는 횟수
    env = RealEnv(step_interval=5, max_steps=50)

    state = env.reset()

    state_dim = len(env.state_cols)
    action_dim = len(env.actions)

    print("[INFO] state_dim:", state_dim)
    print("[INFO] action_dim:", action_dim)
    print("[INFO] model path:", MODEL_PATH)

    if not MODEL_PATH.exists():
        print(f"[ERROR] model not found: {MODEL_PATH}")
        sys.exit(1)

    model = DQN(state_dim, action_dim).to(device)
    #dqn_model.pth 의 모델 parameter만 들고옴. --> 같은 모델은 아니다.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("[INFO] model loaded")
    print("[INFO] real eval start")

    try:
        done = False

        total_reward = 0
        action_counts = [0, 0, 0]

        while not done:
            action, q_values = select_action(model, state)

            action_counts[action] += 1

            next_state, reward, done, info = env.step(action)

            print("=" * 60)
            print(f"step        : {info['step_count']}")
            print(f"q_values    : {q_values}")
            print(f"raw_action  : {info['raw_action']}")
            print(f"safe_action : {info['safe_action']}")
            print(f"action_name : {info['action_name']}")
            print(f"reward      : {reward:.2f}")
            print(f"done        : {done}")
            print(f"state       : {state}")
            print(f"next_state  : {next_state}")

            state = next_state
            total_reward += reward

            print("[INFO] total_reward:", total_reward)
            print("[INFO] action_counts:", action_counts)

    finally:
        print("[INFO] stopping all workers")
        control.stop_all()
        print("[INFO] real eval finished")


if __name__ == "__main__":
    main()
