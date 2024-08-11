import os
import numpy as np
import pandas as pd
from typing import List, Tuple

from imofdma.imofdma import IM_OFDMA
from imofdma.imofdma_fast import IM_OFDMA_FAST
from imofdma import utils
from imofdma import env
from imofdma.logger import get_logger
from imofdma.appconfig import AppConfig

log = get_logger()


def save_user_grouping_option_list(
    grouping_options: List[List[Tuple]], path: str
) -> None:
    """
    Save user grouping options to an Excel file.

    Args:
        grouping_options (List[List[Tuple]]): List of user grouping options.
        path (str): Path to save the Excel file.
    """
    columns = [f"k{i}" for i in range(len(grouping_options[0]))]
    df_groupings = pd.DataFrame(grouping_options, columns=columns)
    df_groupings.index.name = "group_id"
    df_groupings.to_excel(
        path,
        index=True,
        header=columns,
        sheet_name="list",
    )
    log.info(f"User grouping options saved to {path}")


def save_lookup_table(X_table: np.ndarray, path: str) -> None:
    """
    Save lookup table to a text file.

    Args:
        X_table (np.ndarray): The lookup table to be saved.
        path (str): Path to save the text file.
    """
    n_lookup = X_table.shape[-1]
    with open(path, "w") as fp:
        fp.write(f"# Lookup Table Shape: {X_table.shape}\n")
        for ii in range(n_lookup):
            data_slice = X_table[:, :, ii]
            fp.write(f"# Index: {ii}\n")
            np.savetxt(fp, data_slice, fmt="%-7.5f")
    log.info(f"Lookup table saved to {path}")


def setup_simulation() -> AppConfig:
    """
    Set up the simulation environment.

    Returns:
        AppConfig: The configuration object for the simulation.
    """
    cfg_dict = utils.load_yaml(env.PATH_CONFIG)
    cfg = AppConfig.from_dict(cfg_dict)
    cfg.path_simulation = utils.create_simulation_directory()
    env.RND = np.random.RandomState(cfg.sim.random_seed)
    return cfg


def run_simulation(cfg: AppConfig, fast: bool = False) -> None:
    """
    Run the main simulation process.

    Args:
        cfg (AppConfig): The configuration object for the simulation.
    """
    log = get_logger()
    log.info("::: IM-OFDMA Simulation :::")
    log.info(cfg)

    im_ofdma = None
    if fast:
        im_ofdma = IM_OFDMA_FAST(cfg)
    else:
        im_ofdma = IM_OFDMA(cfg)

    path_groupings = os.path.join(cfg.path_simulation, "group_option_list.xlsx")
    save_user_grouping_option_list(im_ofdma.G, path_groupings)

    path_xtable = os.path.join(cfg.path_simulation, "lookup.txt")
    save_lookup_table(im_ofdma.Xtable, path_xtable)

    path_cfg = os.path.join(cfg.path_simulation, "config.yaml")
    utils.save_yaml(cfg.to_dict(), path_cfg)
    log.info(f"Configuration saved to {path_cfg}")

    im_ofdma.run_simulation()


if __name__ == "__main__":
    cfg = setup_simulation()
    run_simulation(cfg, fast=False)
