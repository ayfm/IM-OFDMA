import os
from itertools import permutations, combinations
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from commpy.modulation import PSKModem
from commpy.utilities import dec2bitarray, bitarray2dec
from tqdm import tqdm
from dataclasses import dataclass, field

from . import env
from . import utils
from .imofdma import IM_OFDMA as IM_OFDMA_BASE
from .logger import get_logger
from .appconfig import AppConfig

log = get_logger()


@dataclass
class IM_OFDMA_FAST(IM_OFDMA_BASE):
    """
    Implements the Index Modulation Orthogonal Frequency Division Multiple Access (IM-OFDMA) system.

    This class encapsulates the functionality for creating, transmitting, and receiving
    data using the IM-OFDMA scheme with vectorized operations for improved performance.

    Attributes:
        cfg (AppConfig): Configuration object containing all necessary parameters.
    """

    # additonal attributes
    # iteration counter
    iter: int = field(default=0, init=False)
    # counter for the channel matrix (how many times H is re-sampled)
    H_cnt: int = field(default=0, init=False)
    # channel matrix that is re-sampled last time
    H_current: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        # call the parent class constructor
        super().__post_init__()
        # resample the channel matrix
        self.H_current = self.resample_channel_matrix()

    @property
    def batch_size(self) -> int:
        """Simulation batch size."""
        return self.cfg.sim.batch_size

    def generate_transmission_bits(self) -> np.ndarray:
        """
        Generate bits to be transmitted for multiple users and batches.

        This method creates a 3D array of bits to be transmitted. If a specific transmission
        index is provided in the configuration, it uses that to generate a fixed bit pattern.
        Otherwise, it generates random bits.

        Returns:
            np.ndarray: A 3D array of shape (Nt, nu_N, batch_size) containing the bits to be transmitted.
        """
        transmission_index = self.cfg.sim.x_transmit_idx
        dimensions = (self.Nt, self.nu_N, self.batch_size)

        if transmission_index is None:
            # Generate random bits if no specific index is provided
            transmission_bits = env.RND.randint(0, 2, dimensions)
        else:
            # Generate a fixed bit pattern based on the provided index
            transmission_bits = dec2bitarray(transmission_index, np.prod(dimensions))
            transmission_bits = np.reshape(transmission_bits, dimensions)

        return transmission_bits

    def transmit_Xbits(
        self, Xbits: np.ndarray, H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transmit X bits through the channel for multiple chunks and batches.

        This method simulates the transmission of bits through the IM-OFDMA channel
        using vectorized operations for improved performance.

        Args:
            Xbits (np.ndarray): Input bits to be transmitted, shape (Nk, nu_N, K, batch_size).
            H (np.ndarray): Channel matrix, shape (Lk, Nk, K, batch_size).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - np.ndarray: Received signal Y, shape (Lk, K, batch_size).
                - np.ndarray: Indices of transmitted symbols in the lookup table, shape (K, batch_size).
        """
        Xbits_shaped = Xbits.reshape(-1, self.K, self.batch_size)
        X_indices = np.apply_along_axis(bitarray2dec, 0, Xbits_shaped)

        X = self.Xtable[:, :, X_indices]

        yk_ref = np.einsum("lnkb,nmkb->lmkb", H, X)
        Y = np.diagonal(yk_ref, axis1=0, axis2=1).transpose((2, 0, 1))

        return Y.round(env.PRECISION), X_indices

    def receive_Xbits(
        self, Yw: np.ndarray, H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Receive and process X bits for multiple chunks and batches.

        This method simulates the reception and processing of the transmitted signal.
        It estimates the original bits for each chunk and computes additional metrics.

        Args:
            Yw (np.ndarray): Received signal with noise, shape (Lk, K, batch_size).
            H (np.ndarray): Channel matrix, shape (Lk, Nk, K, batch_size).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - np.ndarray: Estimated bits, shape (Nk, nu_N, K, batch_size).
                - np.ndarray: Indices of estimated symbols in the lookup table, shape (K, batch_size).
                - np.ndarray: Residuals for each chunk and batch, shape (n_lookup, K, batch_size).
                - np.ndarray: Cosine similarities for each chunk and batch, shape (n_lookup, K, batch_size).
        """
        y_ref = np.tensordot(H, self.Xtable, axes=([1], [0])).transpose(0, 3, 4, 1, 2)
        y_ref_diag = np.diagonal(y_ref, axis1=0, axis2=1).transpose(0, 3, 1, 2)

        residuals = np.sum(np.abs(Yw[np.newaxis, :, :, :] - y_ref_diag) ** 2, axis=1)

        nums = np.einsum("slkb,lkb->skb", y_ref_diag, Yw.conjugate())
        norm1 = np.linalg.norm(Yw, axis=0)
        norm2 = np.linalg.norm(y_ref_diag, axis=1)
        denoms = norm1 * norm2
        corrs = nums / denoms

        min_residual_idx = np.argmin(residuals, axis=0)
        Xhbits = np.apply_along_axis(
            lambda idx: np.array(dec2bitarray(idx, self.nu_K)).reshape(
                self.Nk, self.nu_N
            ),
            axis=0,
            arr=min_residual_idx[np.newaxis, :, :],
        )

        return (
            Xhbits,
            min_residual_idx,
            residuals,
            corrs,
        )

    def generate_awgn(self) -> np.ndarray:
        """
        Generate Additive White Gaussian Noise (AWGN) for multiple chunks and batches.

        This method creates complex Gaussian noise with zero mean and specified variance.

        Returns:
            np.ndarray: Complex noise matrix of shape (Lk, K, batch_size).
        """
        dimensions = (self.Lk, self.K, self.batch_size)
        W = np.sqrt(self.var_noise / 2) * (
            env.RND.randn(*dimensions) + 1j * env.RND.randn(*dimensions)
        )
        return W.round(env.PRECISION)

    def generate_channel_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate channel matrices for multiple batches.

        This method creates a list of channel matrices to be used in the next batch,
        resampling when necessary based on the channel coherence time.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - np.ndarray: Channel matrices of shape (Lt, Nt, batch_size).
                - np.ndarray: Channel matrix counters of shape (batch_size,).
        """
        H_list = []
        H_cnt_list = []
        for ii in range(self.iter, self.iter + self.batch_size):
            if ii % self.T == 0:
                self.resample_channel_matrix()
            H_list.append(self.H_current)
            H_cnt_list.append(self.H_cnt)

        H_batch = np.stack(H_list, axis=-1)
        H_cnt_batch = np.array(H_cnt_list)

        return H_batch, H_cnt_batch

    def run_simulation(self) -> None:
        """
        Run the main simulation loop.

        This method executes the entire simulation process, including data generation,
        transmission, reception, and result saving. It uses vectorized operations for
        improved performance and processes data in batches.
        """
        try:
            file_cnt = 1
            data_list = []

            n_iters = self.cfg.sim.n_iterations // self.batch_size
            n_sub_iters = len(self.G)
            n_iters_checkpoint = int(
                np.floor((n_iters / 100) * self.cfg.sim.checkpoint_percent)
            )

            log.info(
                f"Simulation started with {n_iters} iterations (with {n_sub_iters} sub-iterations) and {self.batch_size} batches."
            )

            for ii in tqdm(
                range(1, n_iters + 1), desc="Simulation", unit="iter", colour="#37B6BD"
            ):
                self.iter = ii

                Xbits_batch = self.generate_transmission_bits()
                W_batch = self.generate_awgn()
                H_batch, H_cnt_batch = self.generate_channel_matrix()

                for group_id, user_groups in enumerate(self.G):
                    rows = self.test_grouping_option(
                        Xbits_batch,
                        H_batch,
                        W_batch,
                        group_id,
                        user_groups,
                        iters_H=H_cnt_batch,
                    )
                    data_list.extend(rows)

                if (ii % n_iters_checkpoint == 0) or (ii == n_iters):
                    IM_OFDMA_FAST.save_simulation_data(
                        data_list, self.cfg.path_simulation, file_cnt
                    )
                    file_cnt += 1
                    data_list = []

        except Exception as e:
            log.exception(f"Error occurred: {e}")
        finally:
            log.warning("Simulation ended.")

    def test_grouping_option(
        self,
        Xbits_batch: np.ndarray,
        H_batch: np.ndarray,
        W_batch: np.ndarray,
        group_id: int,
        user_groups: List[Tuple[int, ...]],
        iters_H: np.ndarray,
    ) -> List[Dict]:
        """
        Test a specific grouping option for multiple batches.

        This method simulates transmission and reception for a given user grouping,
        and collects performance metrics for each batch.

        Args:
            Xbits_batch (np.ndarray): Input bits for all users and batches.
            H_batch (np.ndarray): Channel matrix for all batches.
            W_batch (np.ndarray): Noise matrix for all batches.
            group_id (int): ID of the current group configuration.
            user_groups (List[Tuple[int, ...]]): List of user groups.
            iters_H (np.ndarray): Channel matrix iterations for each batch.

        Returns:
            List[Dict]: List of dictionaries containing performance metrics for each chunk and batch.
        """
        # get user indices in the group
        idx_users = list(np.array(user_groups).flatten())
        # get bits for current grouping
        Xgbits = Xbits_batch[idx_users, :, :]
        # get channel matrix for current grouping
        Hg = H_batch[:, idx_users, :]

        # It is better if we reshape Xgbits and Hg to make the chunk axis the last axis
        Hg3 = list()
        Xgbits3 = list()
        for k in range(self.K):
            Hk = Hg[k * self.Lk : (k + 1) * self.Lk, k * self.Nk : (k + 1) * self.Nk, :]
            Hg3.append(Hk)
            Xk = Xgbits[k * self.Nk : (k + 1) * self.Nk, :, :]
            Xgbits3.append(Xk)
        Hg3 = np.stack(Hg3, axis=-2)
        Xgbits3 = np.stack(Xgbits3, axis=-2)

        # transmit the data
        Y, X_indices = self.transmit_Xbits(Xgbits3, Hg3)
        # add white noise to the received signal
        Yw = (Y + W_batch).round(env.PRECISION)

        # receive the data
        (
            Xhbits,
            Xhat_indices,
            residuals,
            corrs,
        ) = self.receive_Xbits(Yw, Hg3)

        # calculate statistics of residuals
        residuals_mean = np.mean(residuals, axis=0).round(env.PRECISION)
        residuals_std = np.std(residuals, axis=0).round(env.PRECISION)
        # calculate statistics of corrs (real-part)
        corrs_real_mean = np.mean(corrs.real, axis=0).round(env.PRECISION)
        corrs_real_std = np.std(corrs.real, axis=0).round(env.PRECISION)
        # calculate statistics of corrs (imaginary-part)
        corrs_imag_mean = np.mean(corrs.imag, axis=0).round(env.PRECISION)
        corrs_imag_std = np.std(corrs.imag, axis=0).round(env.PRECISION)
        # calculate statistics of corrs (magnitude)
        corrs_mag_mean = np.mean(np.abs(corrs), axis=0).round(env.PRECISION)
        corrs_mag_std = np.std(np.abs(corrs), axis=0).round(env.PRECISION)
        # calculate statistics of corrs (angle)
        corrs_ang_mean = np.mean(np.angle(corrs), axis=0).round(env.PRECISION)
        corrs_ang_std = np.std(np.angle(corrs), axis=0).round(env.PRECISION)

        row_list = list()
        # for each chunk, calculate the bitwise error and save the data
        for b in range(self.batch_size):
            for k in range(self.K):
                bitwise_err = IM_OFDMA_FAST.calculate_bitwise_error(
                    Xgbits3[:, :, k, b].flatten(),
                    Xhbits[:, :, k, b].flatten(),
                )
                bitwise_err = bitwise_err / (self.Nk * self.nu_N)

                row = {
                    "iter": (self.iter - 1) * self.batch_size
                    + b,  # simulation iteration
                    "hcnt": iters_H[b],  # H matrix counter
                    "g": group_id,  # grouping id
                    "k": k,  # chunk id
                    "x_idx": X_indices[k, b],  # index of the transmitted data
                    "xh_idx": Xhat_indices[k, b],  # index of the received data
                    "ber": bitwise_err,  # bitwise error rate
                    "residual_mean": residuals_mean[k, b],  # residual for each X value
                    "residual_std": residuals_std[
                        k, b
                    ],  # residual std for each X value
                    "corr_real_mean": corrs_real_mean[
                        k, b
                    ],  # correlation coeff for each X value
                    "corr_real_std": corrs_real_std[
                        k, b
                    ],  # correlation coeff variance for each X value
                    "corr_imag_mean": corrs_imag_mean[
                        k, b
                    ],  # correlation coeff for each X value
                    "corr_imag_std": corrs_imag_std[
                        k, b
                    ],  # correlation coeff variance for each X value
                    "corr_mag_mean": corrs_mag_mean[
                        k, b
                    ],  # correlation coeff for each X value
                    "corr_mag_std": corrs_mag_std[
                        k, b
                    ],  # correlation coeff variance for each X value
                    "corr_ang_mean": corrs_ang_mean[
                        k, b
                    ],  # correlation coeff for each X value
                    "corr_ang_std": corrs_ang_std[
                        k, b
                    ],  # correlation coeff variance for each X value
                }

                # get H matrix for corresponding chunk
                Hk = Hg3[:, :, k, b]

                # add H matrix to the row
                row.update(
                    {
                        f"h{i}{j}": Hk[i, j]
                        for i in range(self.Lk)
                        for j in range(self.Nk)
                    }
                )
                # add Y without noise to the row
                row.update({f"y{i}": Y[i, k, b] for i in range(self.Lk)})
                # add W to the row
                row.update({f"w{i}": W_batch[i, k, b] for i in range(self.Lk)})
                # add Y with noise to the row
                row.update({f"yw{i}": Yw[i, k, b] for i in range(self.Lk)})
                # add the row to the list
                row_list.append(row)

        return row_list

    @staticmethod
    def save_simulation_data(rows: List[Dict], path: str, file_id: int) -> None:
        """
        Save simulation data to a file.

        This function saves the collected simulation data to a file, including
        a small sample in Excel format for debugging purposes.

        Args:
            rows (List[Dict]): List of dictionaries containing simulation results.
            path (str): Base path for saving the data.
            file_id (int): Identifier for the current data file.
        """
        if not rows:
            return

        filename = f"data{file_id}"
        log.info(f"Saving data: {filename}")

        path_data = os.path.join(path, "data", filename)
        os.makedirs(os.path.dirname(path_data), exist_ok=True)

        df_data = pd.DataFrame(rows)
        df_data.sort_values(by=["iter", "g", "k"], inplace=True)
        df_data.reset_index(drop=True, inplace=True)

        utils.save_data(df_data, path_data)

        n_samples = int(len(df_data) * 0.01)
        if n_samples > 0:
            utils.save_data(df_data.loc[:n_samples], path_data + "_sample", fmt=".xlsx")
        log.info("Data saved.")
