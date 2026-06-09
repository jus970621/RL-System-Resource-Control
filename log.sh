#!/bin/bash

BASE_DIR="/mnt/RL_state_expert"

echo "Logging started at $(date)"

start_time=$(date +%s)
duration=$((7*24*60*60))  # 7일

current_date=$(date +%F)

# stress 기준
stress_start_time=$start_time
stress_interval=$((3*24*60*60))

# 초기 디렉토리 생성
MEM_DIR="$BASE_DIR/meminfo/$current_date"
VMSTAT_DIR="$BASE_DIR/vmstat/$current_date"
CPU_DIR="$BASE_DIR/cpu/$current_date"
TEMP_DIR="$BASE_DIR/temp/$current_date"

mkdir -p "$MEM_DIR" "$VMSTAT_DIR" "$CPU_DIR" "$TEMP_DIR"

MEM_LOG="$MEM_DIR/mem.log"
VMSTAT_LOG="$VMSTAT_DIR/vmstat.log"
CPU_LOG="$CPU_DIR/cpu.log"
TEMP_LOG="$TEMP_DIR/temp.log"

# 백그라운드 로그 시작
vmstat 1 >> "$VMSTAT_LOG" &
VMSTAT_PID=$!

mpstat 1 >> "$CPU_LOG" &
CPU_PID=$!

# 온도는 while로 직접 수집
(
while true; do
    echo "===== $(date) =====" >> "$TEMP_LOG"
    sensors >> "$TEMP_LOG"
    sleep 1
done
) &
TEMP_PID=$!

while true; do
    now=$(date +%s)
    elapsed=$((now - start_time))

    # 종료 조건
    if [ "$elapsed" -ge "$duration" ]; then
        echo "Duration reached. Stopping..."
        kill $VMSTAT_PID
        kill $CPU_PID
        kill $TEMP_PID
        break
    fi

    new_date=$(date +%F)

    # 날짜 변경 처리
    if [ "$new_date" != "$current_date" ]; then
        echo "Date changed: $current_date -> $new_date"

        # 기존 프로세스 종료
        kill $VMSTAT_PID
        kill $CPU_PID
        kill $TEMP_PID

        current_date=$new_date

        MEM_DIR="$BASE_DIR/meminfo/$current_date"
        VMSTAT_DIR="$BASE_DIR/vmstat/$current_date"
        CPU_DIR="$BASE_DIR/cpu/$current_date"
        TEMP_DIR="$BASE_DIR/temp/$current_date"

        mkdir -p "$MEM_DIR" "$VMSTAT_DIR" "$CPU_DIR" "$TEMP_DIR"

        MEM_LOG="$MEM_DIR/mem.log"
        VMSTAT_LOG="$VMSTAT_DIR/vmstat.log"
        CPU_LOG="$CPU_DIR/cpu.log"
        TEMP_LOG="$TEMP_DIR/temp.log"

        # 다시 시작
        vmstat 1 >> "$VMSTAT_LOG" &
        VMSTAT_PID=$!

        mpstat 1 >> "$CPU_LOG" &
        CPU_PID=$!

        (
        while true; do
            echo "===== $(date) =====" >> "$TEMP_LOG"
            sensors >> "$TEMP_LOG"
            sleep 1
        done
        ) &
        TEMP_PID=$!
    fi

    # 🔥 3일마다 stress + 랜덤 지연
    stress_elapsed=$((now - stress_start_time))
    if [ "$stress_elapsed" -ge "$stress_interval" ]; then
        delay=$((RANDOM % 300))

        echo "Stress scheduled at $(date), delay ${delay}s" >> "$MEM_LOG"
        sleep $delay

        echo "Running stress at $(date)" >> "$MEM_LOG"
        stress --cpu 4 --vm 2 --vm-bytes 1G --timeout 60

        stress_start_time=$(date +%s)
    fi

    # meminfo 로그
    echo "===== $(date) =====" >> "$MEM_LOG"
    cat /proc/meminfo >> "$MEM_LOG"

    sleep 1
done

echo "Logging finished at $(date)"