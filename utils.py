#!/bin/usr/env python3
from pathlib import Path
import re

def extract_date_from_path(file_path: str) -> str:
    import re
    from pathlib import Path

    for part in Path(file_path).parts:
        if re.match(r"\d{4}-\d{2}-\d{2}", part):
            return part
    raise ValueError(f"날짜 못 찾음: {file_path}")

def get_date_folders(base_path: str):
    """
    base_path 아래에서 날짜 폴더 리스트 반환
    """
    dates = []

    for item in Path(base_path).iterdir():
        if item.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", item.name):
            dates.append(item.name)

    return sorted(dates)

def normalize_timestamp(ts):
    import pandas as pd
    return pd.to_datetime(ts).floor("S")

def safe_int(x):
    try:
        return int(x)
    except:
        return None

def debug_print(df, name):
    print(f"=== {name} ===")
    print(df.head())
    print(df.shape)