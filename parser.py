#!/usr/bin/env python3
import re
from pathlib import Path
from datetime import datetime

import pandas as pd


MEM_KEYS = {
    "MemTotal": "mem_total_kb",
    "MemFree": "mem_free_kb",
    "MemAvailable": "mem_available_kb",
    "Buffers": "buffers_kb",
    "Cached": "cached_kb",
    "SwapTotal": "swap_total_kb",
    "SwapFree": "swap_free_kb",
}

VMSTAT_COLS = [
    "vm_r", "vm_b", "vm_swpd", "vm_free", "vm_buff", "vm_cache",
    "vm_si", "vm_so", "vm_bi", "vm_bo", "vm_in", "vm_cs",
    "vm_us", "vm_sy", "vm_id", "vm_wa", "vm_st",
]

CPU_COLS = [
    "cpu_usr", "cpu_nice", "cpu_sys", "cpu_iowait",
    "cpu_irq", "cpu_soft", "cpu_steal", "cpu_guest",
    "cpu_gnice", "cpu_idle",
]

BASIC_MERGE_COLS = [
    "timestamp",
    "mem_total_kb",
    "mem_free_kb",
    "mem_available_kb",
    "buffers_kb",
    "cached_kb",
    "swap_total_kb",
    "swap_free_kb",
    "vm_r",
    "vm_b",
    "vm_swpd",
    "vm_free",
    "vm_buff",
    "vm_cache",
    "vm_si",
    "vm_so",
    "vm_bi",
    "vm_bo",
    "vm_in",
    "vm_cs",
    "vm_us",
    "vm_sy",
    "vm_id",
    "vm_wa",
    "vm_st",
]

EXPERT_MERGE_COLS = [
    "timestamp",
    "mem_total_kb",
    "mem_free_kb",
    "mem_available_kb",
    "buffers_kb",
    "cached_kb",
    "swap_total_kb",
    "swap_free_kb",
    "vm_r",
    "vm_b",
    "vm_swpd",
    "vm_free",
    "vm_buff",
    "vm_cache",
    "vm_si",
    "vm_so",
    "vm_bi",
    "vm_bo",
    "vm_in",
    "vm_cs",
    "vm_us",
    "vm_sy",
    "vm_id",
    "vm_wa",
    "vm_st",
    "cpu_usr",
    "cpu_nice",
    "cpu_sys",
    "cpu_iowait",
    "cpu_irq",
    "cpu_soft",
    "cpu_steal",
    "cpu_guest",
    "cpu_gnice",
    "cpu_idle",
    "package_temp_c",
]

def extract_date_from_path(file_path: str) -> str:
    parts = Path(file_path).parts
    for part in parts:
        if re.match(r"\d{4}-\d{2}-\d{2}", part):
            return part
    raise ValueError(f"날짜 폴더를 찾지 못함: {file_path}")


def parse_mem_timestamp(line: str) -> datetime:
    line = line.strip()
    if not (line.startswith("=====") and line.endswith("=====")):
        raise ValueError(f"잘못된 mem timestamp line: {line}")

    raw = line.replace("=====", "").strip()
    raw = raw.replace(" KST ", " ")

    return datetime.strptime(raw, "%a %b %d %H:%M:%S %Y")


def parse_temp_timestamp(line: str) -> datetime:
    return parse_mem_timestamp(line)


def parse_mem_log(file_path: str) -> pd.DataFrame:
    rows = []
    current = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("=====") and line.endswith("====="):
                if current is not None:
                    rows.append(current)

                current = {
                    "timestamp": parse_mem_timestamp(line),
                    "date": extract_date_from_path(file_path),
                }
                continue

            if current is None:
                continue

            m = re.match(r"^([A-Za-z_()]+):\s+(\d+)\s+kB$", line)
            if not m:
                continue

            key, value = m.group(1), int(m.group(2))
            if key in MEM_KEYS:
                current[MEM_KEYS[key]] = value

    if current is not None:
        rows.append(current)

    df = pd.DataFrame(rows)
    df["sample_idx"] = range(len(df))  # 👈 여기 추가
    return df


def parse_vmstat_log(file_path: str) -> pd.DataFrame:
    rows = []
    log_date = extract_date_from_path(file_path)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("procs") or line.startswith("r "):
                continue

            parts = line.split()
            if len(parts) != 17:
                continue

            row = {"date": log_date}
            for col, value in zip(VMSTAT_COLS, parts):
                row[col] = int(value)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["sample_idx"] = range(len(df))
    return df


def parse_cpu_log(file_path: str) -> pd.DataFrame:
    rows = []
    log_date = extract_date_from_path(file_path)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Linux"):
                continue
            if "%usr" in line and "%idle" in line:
                continue
            if line.startswith("Average:"):
                continue

            parts = line.split()
            if len(parts) < 13:
                continue

            # 예: 12:00:01 AM all 1.00 0.00 ...
            if parts[2] != "all":
                continue

            time_str = f"{log_date} {parts[0]} {parts[1]}"
            timestamp = datetime.strptime(time_str, "%Y-%m-%d %I:%M:%S %p")

            metrics = list(map(float, parts[3:13]))
            row = {"timestamp": timestamp, "date": log_date}
            for col, value in zip(CPU_COLS, metrics):
                row[col] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    df["sample_idx"] = range(len(df))  # 👈 여기 추가
    return df


def parse_temp_log(file_path: str) -> pd.DataFrame:
    rows = []
    current = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("=====") and line.endswith("====="):
                if current is not None:
                    rows.append(current)

                current = {
                    "timestamp": parse_temp_timestamp(line),
                    "date": extract_date_from_path(file_path),
                    "package_temp_c": None,
                }
                continue

            if current is None:
                continue

            m = re.match(r"^Package id \d+:\s+\+?([0-9.]+)°C", line)
            if m:
                current["package_temp_c"] = float(m.group(1))

    if current is not None:
        rows.append(current)

    df = pd.DataFrame(rows)
    df["sample_idx"] = range(len(df))  # 👈 여기 추가
    return df