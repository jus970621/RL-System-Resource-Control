#!/usr/bin/env python3

import os
import pandas as pd
import parser as ps
import utils


memory_base_path = "data/RL_state/meminfo"
vmstat_base_path = "data/RL_state/vmstat"

expert_memory_base_path = "data/RL_state_expert/meminfo"
expert_vmstat_base_path = "data/RL_state_expert/vmstat"
expert_cpu_base_path = "data/RL_state_expert/cpu"
expert_temp_base_path = "data/RL_state_expert/temp"

basic_output_base = "output/basic"
expert_output_base = "output/expert"

dates = utils.get_date_folders("data/RL_state/meminfo")
expert_dates = utils.get_date_folders("data/RL_state_expert/meminfo")

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mem_available_ratio"] = df["mem_available_kb"] / df["mem_total_kb"]
    df["mem_pressure"] = 1 - df["mem_available_ratio"]
    df["cache_ratio"] = df["cached_kb"] / df["mem_total_kb"]
    df["cpu_user"] = df["vm_us"]
    df["cpu_idle"] = df["vm_id"]
    return df

def add_expert_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mem_available_ratio"] = df["mem_available_kb"] / df["mem_total_kb"]
    df["mem_pressure"] = 1 - df["mem_available_ratio"]
    df["cache_ratio"] = df["cached_kb"] / df["mem_total_kb"]
    df["cpu_user"] = df["vm_us"]
    df["cpu_idle"] = df["vm_id"]
    return df


for date in dates:
    mem_log_path = os.path.join(memory_base_path, date, "mem.log")
    vmstat_log_path = os.path.join(vmstat_base_path, date, "vmstat.log")

    mem_df = ps.parse_mem_log(mem_log_path)
    vmstat_df = ps.parse_vmstat_log(vmstat_log_path)

    date_output_dir = os.path.join(basic_output_base, date)
    os.makedirs(date_output_dir, exist_ok=True)

    mem_df.to_csv(os.path.join(date_output_dir, "mem.csv"), index=False)
    vmstat_df.to_csv(os.path.join(date_output_dir, "vmstat.csv"), index=False)

    merged_df = pd.merge(mem_df, vmstat_df, on=["date", "sample_idx"], how="inner")
    merged_df = add_basic_features(merged_df)
    merged_df.dropna()

    merged_df.to_csv(
        os.path.join(date_output_dir, "basic_merged.csv"),
        index=False
    )

    print(date, len(mem_df), len(vmstat_df))
    print(merged_df[["mem_available_ratio", "cache_ratio", "cpu_user", "cpu_idle"]].isna().sum())
    print("basic_merged:", len(merged_df))

for date in expert_dates:
    expert_mem_log_path = os.path.join(expert_memory_base_path, date, "mem.log")
    expert_vmstat_log_path = os.path.join(expert_vmstat_base_path, date, "vmstat.log")
    expert_cpu_log_path = os.path.join(expert_cpu_base_path, date, "cpu.log")
    expert_temp_log_path = os.path.join(expert_temp_base_path, date, "temp.log")

    expert_mem_df = ps.parse_mem_log(expert_mem_log_path)
    expert_vmstat_df = ps.parse_vmstat_log(expert_vmstat_log_path)
    expert_cpu_df = ps.parse_cpu_log(expert_cpu_log_path)
    expert_temp_df = ps.parse_temp_log(expert_temp_log_path)

    date_output_dir = os.path.join(expert_output_base, date)
    os.makedirs(date_output_dir, exist_ok=True)

    expert_mem_df.to_csv(os.path.join(date_output_dir, "mem.csv"), index=False)
    expert_vmstat_df.to_csv(os.path.join(date_output_dir, "vmstat.csv"), index=False)
    expert_cpu_df.to_csv(os.path.join(date_output_dir, "cpu.csv"), index=False)
    expert_temp_df.to_csv(os.path.join(date_output_dir, "temp.csv"), index=False)

    expert_merged_df = pd.merge(
        expert_mem_df,
        expert_vmstat_df,
        on=["date", "sample_idx"],
        how="inner"
    )

    expert_merged_df = pd.merge(
        expert_merged_df,
        expert_cpu_df,
        on=["date", "sample_idx"],
        how="inner"
    )

    expert_merged_df = pd.merge(
        expert_merged_df,
        expert_temp_df,
        on=["date", "sample_idx"],
        how="inner"
    )

    expert_merged_df = add_expert_features(expert_merged_df)
    expert_merged_df.dropna()

    expert_merged_df.to_csv(
        os.path.join(date_output_dir, "expert_merged.csv"),
        index=False
    )

    print(date, len(expert_mem_df), len(expert_vmstat_df), len(expert_cpu_df), len(expert_temp_df))
    print(expert_merged_df[["mem_available_ratio", "cache_ratio", "cpu_user", "cpu_idle", "mem_pressure"]].isna().sum())
    print("expert_merged:", len(expert_merged_df))