#!/usr/bin/env python3
import numpy as np
import pandas as pd

class Env:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()

        self.state_cols = [
            "mem_available_ratio",
            "cache_ratio",
            "cpu_user",
            "cpu_idle",
            "mem_pressure",
        ]

        if "package_temp_c" in self.df.columns:
            self.state_cols.append("package_temp_c")

        self.expert_state_cols = [
            "mem_available_ratio",
            "cache_ratio",
            "cpu_user",
            "cpu_idle",
            "mem_pressure",
            "package_temp_c",
        ]

        self.actions = {
            0: "hold",  # 현재 workload 유지
            1: "decrease",  # workload 감소
            2: "increase",  # workload 증가
        }

        self.idx = 0

    def compute_reward(self, row):
        reward = 0.0

        if row["cpu_idle"] > 95:
            reward -= 1.0
        elif 60 <= row["cpu_idle"] <= 85:
            reward += 1.0

        if row["mem_pressure"] > 0.8:
            reward -= 2.0
        else:
            reward += 0.5

        if "package_temp_c" in row.index:
            if row["package_temp_c"] > 75:
                reward -= 2.0
            elif row["package_temp_c"] < 68:
                reward += 0.5

        return reward

    def compute_action_cost(self, action):
        if action == 0:
            return 0.0
        elif action == 1:
            return -0.1
        elif action == 2:
            return -0.4

    def compute_action_reward(self, action, row):
        has_temp = "package_temp_c" in row.index
        hot = has_temp and row["package_temp_c"] > 72

        if action == 0:
            if 60 <= row["cpu_idle"] <= 85:
                return 0.3
            return 0.0

        elif action == 1:
            if row["mem_pressure"] > 0.7 or hot:
                return 0.7
            return -0.2

        elif action == 2:
            if row["mem_pressure"] > 0.7 or hot:
                return -1.0
            elif row["cpu_idle"] > 90 and row["mem_pressure"] < 0.5:
                return 0.7
            else:
                return -1.0

    def compute_penalty(self, action):
        penalty = 0.0

        # 반복 패널티
        if hasattr(self, "prev_action") and action == self.prev_action:
            penalty -= 0.4

        # increase 누적 패널티
        """
        if action == 2:
            self.increase_count = getattr(self, "increase_count", 0) + 1
            penalty -= 0.05 * self.increase_count
        else:
            self.increase_count = 0
        """

        return penalty


    def update_internal_state(self, action):
        self.prev_action = action

    def step(self, action):
        if self.idx >= len(self.df) - 1:
            return self.reset(), 0.0, True

        self.idx += 1
        row = self.df.iloc[self.idx]

        reward = 0.0

        reward += self.compute_reward(row)  # 상태 평가
        reward += self.compute_action_cost(action)  # 기본 비용
        reward += self.compute_action_reward(action, row)  # 행동 보상
        reward += self.compute_penalty(action)  # 반복/누적 패널티

        self.update_internal_state(action)

        done = self.idx >= len(self.df) - 1
        next_state = self.df[self.state_cols].iloc[self.idx].values.astype(np.float32)

        return next_state, reward, done

    def reset(self):
        self.idx = 0
        self.prev_action = None
        self.increase_count = 0
        return self.df[self.state_cols].iloc[self.idx].values.astype(np.float32)

if __name__=="__main__":
    data_path = "output/expert/2026-04-03/expert_merged.csv"
    data_frame = pd.read_csv(data_path)
    #print(type(data_frame))
    env = Env(data_frame)
    action =env.actions

    for key, value in action.items():
        print(key, '번째:', value)

