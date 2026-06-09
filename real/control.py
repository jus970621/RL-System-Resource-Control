#!/bin/usr/env python3
import subprocess
import sys
from pathlib import Path

PID_FILE = Path("worker_pids.txt")
MAX_WORKERS = 2

BASE_DIR = Path(__file__).resolve().parent
WORKER_PATH = BASE_DIR / "cpu_worker.py"

#프로세스 ID 번호 출력 [1234, 5678, 91011]
def load_pids():
    if not PID_FILE.exists():
        return []
    pid_list = [int(x) for x in PID_FILE.read_text().splitlines() if x.strip().isdigit()]
    #print('pid_list:',pid_list)
    return pid_list


#프로세스 id를 문서파일로 저장
#1234
#5678
#91011 이런식으로 1열로 세로로 저장
def save_pids(pids):
    PID_FILE.write_text("\n".join(map(str, pids)) + ("\n" if pids else ""))
    #print('save_pids', pids)


def is_pid_alive(pid):
    import os
    #linux 용도
    """
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    """
    # tasklist /FI 1234(PID_Number) >> 1234라는 넘버 프로세스 리스트를 알려줘..
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,#capture_output 출력하지 말고 변수에 저장
        text=True #Text로 받음.
    )
    #result.stdout 내용
    #cmd.exe         1234   Console     6,908 K
    #문자열 1234가 있으면 True 출력
    return str(pid) in result.stdout

#지금 현재 남아있는 pid_list를 저장 pid_list를 list로 반환
def cleanup_dead_pids():
    pids = load_pids()
    #print('load_pids:',pids)

    alive = []

    for pid in pids:
        if is_pid_alive(pid):
            #print('pid:',pid)
            alive.append(pid)

    #print('alive_before:',alive)
    save_pids(alive)
    #print('alive_after:',alive)
    return alive


def increase():
    pids = cleanup_dead_pids()

    if len(pids) >= MAX_WORKERS:
        print(f"[INFO] max workers reached: {MAX_WORKERS}")
        return
    #print('sys.executable:', sys.executable)
    #print('WORKER_PATH:',WORKER_PATH)
    proc = subprocess.Popen(
        [sys.executable, str(WORKER_PATH)],
        stdout=subprocess.DEVNULL,#print() 해도 출력 안되게 해줘
        stderr=subprocess.DEVNULL,#Trackback error 출력 안되게 해줘.
    )
    #print('pid_number:',proc.pid)
    pids.append(proc.pid)
    save_pids(pids)
    print(f"[INFO] increase | worker added: PID={proc.pid} | count={len(pids)}")


def decrease():
    pids = cleanup_dead_pids()

    if not pids:
        print("[INFO] no worker to stop")
        return
    #list 첫번째 출력 후 제거
    #[1234, 5678] >> 1234 출력 후 1234process 제거
    #pid = 1234
    #pids = [5678]
    pid = pids.pop(0)

    #출력 된 1234 제거 taskkill /PID 1234 /F
    try:
        import os, signal
        #linux 용도
        #os.kill(pid, signal.SIGTERM)
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(f"[INFO] decrease | worker stopped: PID={pid}")
    except Exception as e:
        print(f"[WARN] failed to stop PID={pid}: {e}")

    save_pids(pids)


def hold():
    pids = cleanup_dead_pids()
    print(f"[INFO] hold | workers={len(pids)}")


def apply_action(action):
    # 기존 프로젝트 기준
    # 0 = hold, 1 = decrease, 2 = increase
    if action == 0:
        hold()
    elif action == 1:
        decrease()
    elif action == 2:
        increase()
    else:
        raise ValueError(f"unknown action: {action}")


def stop_all():
    while cleanup_dead_pids():
        decrease()

def get_worker_count():
    pid_list = load_pids()
    return len(pid_list)

if __name__ == "__main__":
    import time


    while True:
        apply_action(2)
        time.sleep(3)
        apply_action(2)
        time.sleep(3)
        print('pid_list :', load_pids())
        apply_action(1)
        print('pid_list :',load_pids())
        time.sleep(3)
        apply_action(0)
        time.sleep(3)


    #apply_action(2)
    #apply_action(1)
    #apply_action(0)
    #stop_all()

