import os
from datetime import datetime
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import yaml

from . import env
from .logger import get_logger

FILE_FORMATS = [".xlsx", ".h5", ".parquet"]
log = get_logger()


def load_yaml(path: str) -> Dict[str, Any]:
    """Load data from a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Save data to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f, indent=4)


def create_simulation_directory() -> str:
    """Create a new simulation directory with a timestamp."""
    current_time = datetime.now()
    folder_name = current_time.strftime("%d%b%y_%Hh%Mm")
    path_simulation = os.path.join(env.PATH_OUTPUTS, folder_name)

    n_trial = 1
    while os.path.exists(path_simulation):
        n_trial += 1
        if n_trial > 100:
            raise Exception("Too many trials for creating simulation directory.")
        path_simulation = os.path.join(env.PATH_OUTPUTS, f"{folder_name}__{n_trial}")

    os.makedirs(path_simulation)

    path_logs = os.path.join(path_simulation, "simulation.log")
    logger = get_logger(path=path_logs)
    logger.debug("Simulation directory: %s", path_simulation)

    return path_simulation


def save_data(df: pd.DataFrame, path: str, fmt: str = ".parquet") -> None:
    """Save DataFrame to a file with specified format."""
    if not any(path.endswith(f) for f in FILE_FORMATS):
        path += fmt

    if fmt == ".xlsx":
        df.to_excel(path, index=True)
    elif fmt == ".h5":
        df.to_hdf(path, "data", mode="w")
    elif fmt == ".parquet":
        save_data_parquet(df, path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def save_data_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to a Parquet file with data type information."""
    if not path.endswith(".parquet"):
        path += ".parquet"

    df_copy = df.copy()
    parent_dir = os.path.dirname(path)
    path_dtypes = os.path.join(parent_dir, "dtypes.json")

    dtypes = df_copy.dtypes.apply(lambda x: x.name).to_dict()
    with open(path_dtypes, "w") as f:
        json.dump(dtypes, f, indent=3)

    for col in df_copy.columns:
        if df_copy[col].dtype == "complex":
            df_copy[col] = df_copy[col].astype(str)

    df_copy.to_parquet(path, index=True, engine="pyarrow", compression="gzip")


def read_data(path: str, fmt: str = ".parquet") -> pd.DataFrame:
    """Read data from a file with specified format."""
    if not any(path.endswith(f) for f in FILE_FORMATS):
        path += fmt

    if fmt == ".xlsx":
        return pd.read_excel(path, index_col=0)
    elif fmt == ".h5":
        return pd.read_hdf(path, "data")
    elif fmt == ".parquet":
        return read_data_parquet(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def read_data_parquet(path: str) -> pd.DataFrame:
    """Read DataFrame from a Parquet file with data type restoration."""
    if not path.endswith(".parquet"):
        path += ".parquet"

    parent_dir = os.path.dirname(path)
    path_dtypes = os.path.join(parent_dir, "dtypes.json")

    try:
        with open(path_dtypes, "r") as f:
            dtypes = json.load(f)
    except FileNotFoundError:
        dtypes = None
        log.warning("Data types file not found. Using default data types.")

    df = pd.read_parquet(path, engine="pyarrow")

    if dtypes is not None:
        for col, dtype in dtypes.items():
            if dtype.startswith("complex"):
                df[col] = df[col].apply(complex)

    return df
