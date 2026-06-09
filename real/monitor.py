#!/usr/bin/env python3
import psutil
import numpy as np


STATE_COLS = [
    "mem_available_ratio", # available_mem/total_mem
    "cache_ratio", # cache_mem / total_mem
    "cpu_user", # cpu가 사용자 프로그램에 사용하는 비율
    "cpu_idle",# cpu가 놀고 있는 비율
    "mem_pressure",# 1-mem_available_raito
    "package_temp_c",# cpu 온도
]


def get_cpu_temp():
    """
    Windows에서는 대부분 psutil로 온도 수집이 안 될 수 있음.
    실패하면 -1.0 반환.
    """
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return -1.0

        for name, entries in temps.items():
            for entry in entries:
                if entry.current is not None:
                    return float(entry.current)

        return -1.0
    except Exception:
        return -1.0


def get_state_dict():
    cpu_percent = psutil.cpu_percent(interval=1)

    # 0~100 범위 밖 이상값 방지
    cpu_percent = max(0.0, min(cpu_percent, 100.0))

    mem = psutil.virtual_memory()

    mem_available_ratio = mem.available / mem.total
    mem_pressure = 1.0 - mem_available_ratio

    # Windows에서는 cached 값이 없을 수 있음
    cached = getattr(mem, "cached", 0)
    cache_ratio = cached / mem.total if mem.total > 0 else 0.0

    cpu_user = cpu_percent
    cpu_idle = 100.0 - cpu_percent

    package_temp_c = get_cpu_temp()

    return {
        "mem_available_ratio": float(mem_available_ratio),
        "cache_ratio": float(cache_ratio),
        "cpu_user": float(cpu_user),
        "cpu_idle": float(cpu_idle),
        "mem_pressure": float(mem_pressure),
        "package_temp_c": float(package_temp_c),
    }


def get_state():
    state_dict = get_state_dict()
    return np.array([state_dict[col] for col in STATE_COLS], dtype=np.float32)


if __name__ == "__main__":
    """
    while True:
        state_dict = get_state_dict()
        print(state_dict)
    """
    count = 0
    state_dict = get_state_dict()
    print('state 데이터 타입 :',type(state_dict))

    for key, value in state_dict.items():
        print(count,'번째:' , key, value)
        count += 1