#!/usr/bin/env python3
from env_real import RealEnv
import control

def main():
    #stap_interval >> 각 액션별 기다리는 시간 ex) 액션 hold 후 5초 대기 액션 increase 후 5초 대기 액션 decrease
    #max_steps = 한 에피소드(episode는 한번이니 의미 없음)당 학습하는 횟수
    env = RealEnv(step_interval=5, max_steps=50)

    state = env.reset()

    state_dim = len(env.state_cols)
    action_dim = len(env.actions)

    print("[INFO] state_dim:", state_dim)
    print("[INFO] action_dim:", action_dim)

    try:
        done = False

        total_reward = 0
        action_counts = [0, 0, 0]

        while not done:
            cpu_idle = state[3]
            mem_pressure = state[4]

            # rule baseline v1
            if cpu_idle > 90 and mem_pressure < 0.705:
                action = 2
            elif cpu_idle < 80 or mem_pressure > 0.715:
                action = 1
            else:
                action = 0

            action_counts[action] += 1

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            print("=" * 60)
            print(f"step        : {info['step_count']}")
            print(f"raw_action  : {info['raw_action']}")
            print(f"safe_action : {info['safe_action']}")
            print(f"action_name : {info['action_name']}")
            print(f"reward      : {reward:.2f}")
            print(f"done        : {done}")
            print(f"state       : {state}")
            print(f"next_state  : {next_state}")
            print("[INFO] total_reward:", total_reward)
            print("[INFO] action_counts:", action_counts)

            state = next_state

    finally:
        print("[INFO] stopping all workers")
        control.stop_all()
        print("[INFO] real eval finished")

if __name__ == "__main__":
    main()


