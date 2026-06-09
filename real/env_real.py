#!/usr/bin/env python3
import time
import numpy as np

import monitor
import control


class RealEnv:
    def __init__(self, step_interval=3, max_steps=30):
        self.step_interval = step_interval
        self.max_steps = max_steps
        self.step_count = 0

        self.actions = {
            0: "hold",
            1: "decrease",
            2: "increase",
        }

        self.state_cols = monitor.STATE_COLS

    def reset(self):
        self.step_count = 0

        # 시작할 때 worker 정리하고 싶으면 사용
        control.stop_all()

        state = monitor.get_state()
        return state

    def compute_reward(self, state, action, next_state):
        """
        state 순서:
        0 mem_available_ratio
        1 cache_ratio
        2 cpu_user
        3 cpu_idle
        4 mem_pressure
        5 package_temp_c
        """
        mem_available_ratio = next_state[0]
        cpu_user = next_state[2]
        cpu_idle = next_state[3]
        mem_pressure = next_state[4]
        package_temp_c = next_state[5]

        reward = 0.0

        # 1. 상태 기반 reward
        if 60 <= cpu_idle <= 85:
            reward += 1.0
        elif cpu_idle > 90:
            reward -= 0.5
        elif cpu_idle < 40:
            reward -= 1.5

        # 2. mem_pressure 기반 reward
        if mem_pressure > 0.8:
            reward -= 2.0
        elif mem_pressure < 0.72:
            reward += 1.0
        else:
            reward += 0.3

        # 3. 온도 penalty reward
        if package_temp_c > 0:
            if package_temp_c > 80:
                reward -= 5.0
            elif package_temp_c > 75:
                reward -= 2.0

        current_workers = control.get_worker_count()

        # 4. action별 reward
        if action == 0:  # hold
            if cpu_idle > 90 and mem_pressure < 72:
                reward -= 0.7  # 여유 있는데 hold하면 손해
            else:
                reward += 0.0

        elif action == 1:  # decrease
            if  current_workers <= 0:
                reward -= 1.0
            elif cpu_idle < 50 or mem_pressure > 78:
                reward += 1.0  # 위험하면 줄이는 게 좋음
            else:
                reward -= 0.5  # 여유 있는데 줄이면 손해

        elif action == 2: #increase
            if current_workers >= control.MAX_WORKERS:
                reward -= 1.0
            elif cpu_idle > 85 and mem_pressure < 76:
                reward += 2.0
            elif cpu_idle < 60 or mem_pressure > 78:
                reward -= 1.0
            else:
                reward -= 0.1

        return float(reward)

    def safety_filter(self, action, state):
        mem_pressure = state[4]
        package_temp_c = state[5]

        # 메모리 압박 심하면 increase 행동을 hold로 변경
        if action == 2 and mem_pressure > 0.8:
            print("[SAFETY] mem_pressure high -> increase blocked, change to hold")
            return 0

        # cpu 온도가 75도 초과하면 높으면 강제 decrease 행동 변경
        if package_temp_c > 0 and package_temp_c > 75:
            print("[SAFETY] temperature high -> force decrease")
            return 1

        return action

    def is_done(self, next_state):
        mem_pressure = next_state[4]
        package_temp_c = next_state[5]

        if self.step_count >= self.max_steps:
            return True

        if mem_pressure > 0.9:
            print("[DONE] mem_pressure too high")
            return True

        if package_temp_c > 0 and package_temp_c > 85:
            print("[DONE] temperature too high")
            return True

        return False

    def step(self, action):
        self.step_count += 1

        state_before = monitor.get_state()

        safe_action = self.safety_filter(action, state_before)

        control.apply_action(safe_action)

        time.sleep(self.step_interval)

        next_state = monitor.get_state()

        reward = self.compute_reward(
            state=state_before,
            action=safe_action,
            next_state=next_state,
        )

        done = self.is_done(next_state)

        info = {
            "raw_action": action,
            "safe_action": safe_action,
            "action_name": self.actions[safe_action],
            "step_count": self.step_count,
        }

        return next_state, reward, done, info


if __name__ == "__main__":
    env = RealEnv(step_interval=3, max_steps=10)

    state = env.reset()
    print("[RESET]", state)

    #test_actions = [2, 0, 2, 0, 1, 0, 1, 0]
    action = 0

    next_state, reward, done, info = env.step(action)

    print("=" * 60)
    print("info:", info)
    print("next_state:", next_state)
    print("reward:", reward)
    print("done:", done)


    """
    for action in test_actions:
        next_state, reward, done, info = env.step(action)

        print("=" * 60)
        print("info:", info)
        print("next_state:", next_state)
        print("reward:", reward)
        print("done:", done)

        if done:
            break

    control.stop_all()
    print("[END] stop_all")
    """