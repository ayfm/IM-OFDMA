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
from .logger import get_logger
from .appconfig import AppConfig

log = get_logger()


@dataclass
class IM_OFDMA:
    """
    Implements the Index Modulation Orthogonal Frequency Division Multiple Access (IM-OFDMA) system.

    This class encapsulates the functionality for creating, transmitting, and receiving
    data using the IM-OFDMA scheme. It includes methods for generating lookup tables,
    user grouping, and simulating the transmission process.

    Attributes:
        cfg (AppConfig): Configuration object containing all necessary parameters.
        _psk_modem (PSKModem): PSK modulation object for signal modulation.
        Xtable (np.ndarray): Lookup table for IM-OFDMA modulation.
        G (List[List[Tuple[int, ...]]]): List of user grouping options.
    """

    cfg: "AppConfig"
    _psk_modem: PSKModem = field(init=False)
    Xtable: np.ndarray = field(init=False)
    G: List[List[Tuple[int, ...]]] = field(init=False)

    def __post_init__(self):
        """
        Initialize the IM_OFDMA object after __init__.

        This method sets up the PSK modem, creates the lookup table, and generates
        user grouping options.
        """
        self._psk_modem = PSKModem(self.M)
        self.Xtable = self.create_lookup_table()
        self.G = self.create_user_grouping_options()

    @property
    def M(self) -> int:
        """Modulation order."""
        return self.cfg.im_ofdma.M

    @property
    def K(self) -> int:
        """Number of chunks."""
        return self.cfg.im_ofdma.K

    @property
    def Nt(self) -> int:
        """Total number of users."""
        return self.cfg.im_ofdma.n_total_users

    @property
    def Nk(self) -> int:
        """Number of users per chunk."""
        return self.cfg.im_ofdma.Nk

    @property
    def Lt(self) -> int:
        """Total number of subcarriers."""
        return self.cfg.im_ofdma.n_total_subcarriers

    @property
    def Lk(self) -> int:
        """Number of subcarriers per chunk."""
        return self.cfg.im_ofdma.Lk

    @property
    def nu_M(self) -> int:
        """Number of bits for modulation."""
        return self.cfg.im_ofdma.nu_L

    @property
    def nu_K(self) -> int:
        """Number of bits per chunk."""
        return self.cfg.im_ofdma.nu_K

    @property
    def nu_L(self) -> int:
        """Number of bits for subcarrier selection."""
        return self.cfg.im_ofdma.nu_L

    @property
    def nu_N(self) -> int:
        """Total number of bits per user."""
        return self.cfg.im_ofdma.nu_N

    @property
    def n_lookup(self) -> int:
        """Size of the lookup table."""
        return self.cfg.im_ofdma.n_lookup

    @property
    def var_channel(self) -> float:
        """Variance of the channel coefficient distribution."""
        return self.cfg.im_ofdma.var_H

    @property
    def var_noise(self) -> float:
        """Variance of the noise."""
        return self.cfg.im_ofdma.var_W

    @property
    def T(self) -> int:
        """Channel coherence time."""
        return self.cfg.im_ofdma.T

    def create_lookup_table(self) -> np.ndarray:
        """
        Generate the lookup table for IM-OFDMA modulation.

        This method creates a 3D lookup table where each entry represents a possible
        transmission state for a user in a chunk. The table is indexed by user, subcarrier,
        and lookup index.

        Returns:
            np.ndarray: A 3D array of shape (Nk, Lk, n_lookup) containing complex modulation symbols.
        """
        log.info("Generating Lookup Table...")
        X_table = np.zeros((self.Nk, self.Lk, self.n_lookup), dtype=np.complex64)

        for t in range(self.n_lookup):
            log.debug(f"Processing lookup index: {t}")
            Sym = dec2bitarray(t, self.nu_K)  # Convert lookup index to binary

            for uid in range(self.Nk):
                log.debug(f"Processing user ID: {uid}")
                Svec = np.zeros(self.Lk, dtype=np.complex64)
                Sym_n = Sym[
                    uid * self.nu_N : (uid + 1) * self.nu_N
                ]  # Extract bits for current user
                log.debug(f"User symbol: {Sym_n}")

                l_bits, m_bits = (
                    Sym_n[: self.nu_L],
                    Sym_n[self.nu_L :],
                )  # Split bits for subcarrier selection and modulation
                ch_ind = bitarray2dec(
                    l_bits
                )  # Convert subcarrier selection bits to decimal
                log.debug(f"Selected subcarrier index: {ch_ind}")

                # Modulate the symbol or use default for M=1
                mod_sym = (
                    self._psk_modem.modulate(m_bits)
                    if self.M != 1
                    else (1 + 1j) / np.sqrt(2)
                )
                log.debug(f"Modulated symbol: {mod_sym}")

                Svec[ch_ind] = (
                    mod_sym  # Place modulated symbol in the selected subcarrier
                )
                X_table[uid, :, t] = Svec

        log.info(f"Lookup Table Shape: {X_table.shape}")
        return X_table.round(env.PRECISION)

    def create_user_grouping_options(self) -> List[List[Tuple[int, ...]]]:
        """
        Generate all valid combinations of users into groups.

        This method creates all possible ways to divide the total number of users into
        K groups of equal size. It ensures that each user appears in exactly one group.

        Returns:
            List[List[Tuple[int, ...]]]: A list of valid user groupings, where each grouping
            is a list of K tuples, and each tuple contains the user indices for that group.

        Raises:
            ValueError: If the total number of users is not divisible by the number of groups.
        """
        log.info("Generating User Grouping Options...")
        group_size = self.Nt // self.K
        if self.Nt % self.K != 0:
            raise ValueError(
                f"Total users ({self.Nt}) should be divisible by number of groups ({self.K})."
            )

        users = list(range(self.Nt))
        groups = list(combinations(users, group_size))

        valid_groupings = [
            perm
            for perm in permutations(groups, self.K)
            if len(set(sum(perm, ())))
            == self.Nt  # Ensure each user appears exactly once
        ]

        log.info(f"Number of Grouping Options: {len(valid_groupings)}")
        return valid_groupings

    def generate_transmission_bits(self) -> np.ndarray:
        """
        Generate bits to be transmitted.

        This method creates a 2D array of bits to be transmitted. If a specific transmission
        index is provided in the configuration, it uses that to generate a fixed bit pattern.
        Otherwise, it generates random bits.

        Returns:
            np.ndarray: A 2D array of shape (Nt, nu_N) containing the bits to be transmitted.
        """
        x_transmit_idx = self.cfg.sim.x_transmit_idx

        if x_transmit_idx is None:
            # Generate random bits if no specific index is provided
            X_bits = env.RND.randint(0, 2, (self.Nt, self.nu_N))
        else:
            # Generate a fixed bit pattern based on the provided index
            X_bits = dec2bitarray(x_transmit_idx, (self.Nt * self.nu_N))
            X_bits = np.reshape(X_bits, (self.Nt, self.nu_N))

        return X_bits

    def transmit_Xbits(
        self, Xbits: np.ndarray, H: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Transmit X bits through the channel.

        This method simulates the transmission of bits through the IM-OFDMA channel.
        It processes each chunk separately, applying the channel matrix and selecting
        the appropriate symbols from the lookup table.

        Args:
            Xbits (np.ndarray): Input bits to be transmitted, shape (Nt, nu_N).
            H (np.ndarray): Channel matrix, shape (Lt, Nt).

        Returns:
            Tuple[np.ndarray, List[int]]:
                - np.ndarray: Received signal Y, shape (Lk, K).
                - List[int]: Indices of transmitted symbols in the lookup table.
        """
        Y = []
        X_indices = []

        for k in range(self.K):
            # Extract channel matrix for current chunk
            Hk = H[k * self.Lk : (k + 1) * self.Lk, k * self.Nk : (k + 1) * self.Nk]
            # Extract bits for current chunk
            Xk = Xbits[k * self.Nk : (k + 1) * self.Nk, :]
            # Convert bits to lookup table index
            idx = bitarray2dec(Xk.flatten())
            X_indices.append(idx)
            # Get symbol from lookup table
            X = self.Xtable[:, :, idx]
            # Apply channel and extract diagonal (received signal for this chunk)
            yk = np.diag(np.dot(Hk, X))
            Y.append(yk)

        Y = np.stack(Y, axis=1).round(env.PRECISION)
        return Y, X_indices

    def receive_Xbits(
        self, Yw: np.ndarray, H: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Receive and process X bits.

        This method simulates the reception and processing of the transmitted signal.
        It estimates the original bits for each chunk and computes additional metrics.

        Args:
            Yw (np.ndarray): Received signal with noise, shape (Lk, K).
            H (np.ndarray): Channel matrix, shape (Lt, Nt).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - np.ndarray: Estimated bits, shape (Nt, nu_N).
                - np.ndarray: Indices of estimated symbols in the lookup table.
                - np.ndarray: Residuals for each possible symbol, shape (n_lookup, K).
                - np.ndarray: Correlation coefficients for each possible symbol, shape (n_lookup, K).
        """
        Xhat_bits, Xh_indices, residuals, corrs = [], [], [], []

        for k in range(self.K):
            yk = Yw[:, k]
            Hk = H[k * self.Lk : (k + 1) * self.Lk, k * self.Nk : (k + 1) * self.Nk]
            Xhbits, xh_idx, residual, csim = self.estimate_Xbits(yk, Hk)
            Xhat_bits.append(Xhbits)
            residuals.append(residual)
            Xh_indices.append(xh_idx)
            corrs.append(csim)

        Xhat_bits = np.array(Xhat_bits).reshape(self.Nt, -1)
        residuals = np.stack(residuals, axis=1)
        Xh_indices = np.array(Xh_indices)
        corrs = np.stack(corrs, axis=1)

        return Xhat_bits, Xh_indices, residuals, corrs

    def estimate_Xbits(
        self, yk: np.ndarray, Hk: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Estimate X bits from received signal.

        This method estimates the original transmitted bits for a single chunk
        by comparing the received signal with all possible symbols in the lookup table.

        Args:
            yk (np.ndarray): Received signal for one chunk, shape (Lk,).
            Hk (np.ndarray): Channel matrix for one chunk, shape (Lk, Nk).

        Returns:
            Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
                - np.ndarray: Estimated bits, shape (Nk, nu_N).
                - int: Index of the estimated symbol in the lookup table.
                - np.ndarray: Residuals for each possible symbol, shape (n_lookup,).
                - np.ndarray: Correlation coefficients for each possible symbol, shape (n_lookup,).
        """
        residuals = np.zeros(self.n_lookup)
        corrs = np.zeros(self.n_lookup, dtype=np.complex64)

        for ii in range(self.n_lookup):
            X = self.Xtable[:, :, ii]
            y_ref = np.diag(np.dot(Hk, X))
            residuals[ii] = np.linalg.norm(yk - y_ref) ** 2
            corrs[ii] = IM_OFDMA.correlation_coefficient(yk, y_ref)

        min_residual_idx = np.argmin(residuals)
        Xhbits = dec2bitarray(min_residual_idx, self.nu_K)
        Xhbits = Xhbits.reshape(self.Nk, self.nu_N)

        return (
            Xhbits,
            min_residual_idx,
            residuals.round(env.PRECISION),
            corrs.round(env.PRECISION),
        )

    def generate_awgn(self) -> np.ndarray:
        """
        Generate Additive White Gaussian Noise (AWGN).

        This method creates complex Gaussian noise with zero mean and specified variance.

        Returns:
            np.ndarray: Complex noise matrix of shape (Lk, K).
        """
        dimensions = (self.Lk, self.K)
        # Generate complex noise: real and imaginary parts are independent Gaussian
        W = np.sqrt(self.var_noise / 2) * (
            env.RND.randn(*dimensions) + 1j * env.RND.randn(*dimensions)
        )
        return W.round(env.PRECISION)

    def resample_channel_matrix(self) -> np.ndarray:
        """
        Re-sample the channel matrix.

        This method generates a new complex Gaussian channel matrix.

        Returns:
            np.ndarray: Complex channel matrix of shape (Lt, Nt).
        """
        dimensions = (self.Lt, self.Nt)
        # Generate complex channel coefficients: real and imaginary parts are independent Gaussian
        H = np.sqrt(self.var_channel / 2) * (
            env.RND.randn(*dimensions) + 1j * env.RND.randn(*dimensions)
        )
        return H.round(env.PRECISION)

    def run_simulation(self) -> None:
        """
        Run the main simulation loop.

        This method executes the entire simulation process, including data generation,
        transmission, reception, and result saving.
        """
        try:
            file_cnt = 1
            data_list = []
            H = self.resample_channel_matrix()
            hh = 0

            n_iters = self.cfg.sim.n_iterations
            n_sub_iters = len(self.G)
            n_iters_checkpoint = int(
                np.floor((n_iters / 100) * self.cfg.sim.checkpoint_percent)
            )

            log.info(
                f"Simulation started with {n_iters} iterations (with {n_sub_iters} sub-iterations)"
            )

            for ii in tqdm(
                range(1, n_iters + 1), desc="Simulation", unit="iter", colour="#37B6BD"
            ):
                # Resample channel matrix every T iterations
                if (ii - 1) % self.T == 0:
                    H = self.resample_channel_matrix()
                    hh += 1
                    log.debug(f"Hmat shape: {H.shape}")

                Xbits = self.generate_transmission_bits()
                W = self.generate_awgn()

                # Test each user grouping option
                for group_id, user_groups in enumerate(self.G):
                    rows = self.test_grouping_option(
                        Xbits, H, W, group_id, user_groups, sim_iter=ii, h_iter=hh
                    )
                    data_list.extend(rows)

                # Save data at checkpoints or at the end of simulation
                if (ii % n_iters_checkpoint == 0) or (ii == n_iters):
                    IM_OFDMA.save_simulation_data(
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
        Xbits: np.ndarray,
        H: np.ndarray,
        W: np.ndarray,
        group_id: int,
        user_groups: List[Tuple[int, ...]],
        sim_iter: int,
        h_iter: int,
    ) -> List[Dict]:
        """
        Test a specific grouping option.

        This method simulates transmission and reception for a given user grouping,
        and collects performance metrics.

        Args:
            Xbits (np.ndarray): Input bits for all users.
            H (np.ndarray): Channel matrix.
            W (np.ndarray): Noise matrix.
            group_id (int): ID of the current group configuration.
            user_groups (List[Tuple[int, ...]]): List of user groups.
            sim_iter (int): Current simulation iteration.
            h_iter (int): Current channel matrix iteration.

        Returns:
            List[Dict]: List of dictionaries containing performance metrics for each chunk.
        """
        idx_users = list(np.array(user_groups).flatten())
        Xgbits = Xbits[idx_users, :]
        Hg = H[:, idx_users]

        Y, X_indices = self.transmit_Xbits(Xgbits, Hg)
        Yw = (Y + W).round(env.PRECISION)
        Xhbits, Xhat_indices, residuals, corrs = self.receive_Xbits(Yw, Hg)

        row_list = []
        for k in range(self.K):
            bitwise_err = IM_OFDMA.calculate_bitwise_error(
                Xgbits[k * self.Nk : (k + 1) * self.Nk, :].flatten(),
                Xhbits[k * self.Nk : (k + 1) * self.Nk, :].flatten(),
            )
            bitwise_err = bitwise_err / (self.Nk * self.nu_N)

            Hk = Hg[k * self.Lk : (k + 1) * self.Lk, k * self.Nk : (k + 1) * self.Nk]

            row = {
                "iter": sim_iter,
                "hcnt": h_iter,
                "g": group_id,
                "k": k,
                "x_idx": X_indices[k],
                "xh_idx": Xhat_indices[k],
                "ber": bitwise_err,
                "residual": residuals[:, k],
                "corr": corrs[:, k],
                **{
                    f"h{i}{j}": Hk[i, j] for i in range(self.Lk) for j in range(self.Nk)
                },
                **{f"y{i}": Y[i, k] for i in range(self.Lk)},
                **{f"w{i}": W[i, k] for i in range(self.Lk)},
                **{f"yw{i}": Yw[i, k] for i in range(self.Lk)},
            }
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
        # Convert list-like columns to strings for storage
        df_data["residual"] = df_data["residual"].apply(lambda x: ";".join(map(str, x)))
        df_data["corr"] = df_data["corr"].apply(lambda x: ";".join(map(str, x)))

        utils.save_data(df_data, path_data)

        # Save a small sample in Excel format for debugging
        n_samples = int(len(df_data) * 0.01)
        if n_samples > 0:
            utils.save_data(df_data.loc[:n_samples], path_data + "_sample", fmt=".xlsx")
        log.info("Data saved.")

    @staticmethod
    def calculate_bitwise_error(x1: np.ndarray, x2: np.ndarray) -> int:
        """
        Calculate the number of bit errors between two bit arrays.

        This function computes the Hamming distance between two binary arrays,
        which represents the number of positions at which the corresponding bits
        are different.

        Args:
            x1 (np.ndarray): First bit array.
            x2 (np.ndarray): Second bit array.

        Returns:
            int: The number of positions where the bits differ.

        Raises:
            ValueError: If the input arrays have different shapes.

        Note:
            The input arrays should contain only binary values (0 or 1).
        """
        if x1.shape != x2.shape:
            raise ValueError("Input arrays must have the same shape")
        return np.sum(np.bitwise_xor(x1, x2))

    @staticmethod
    def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> complex:
        """
        Calculate complex correlation coefficient between two complex vectors.

        The coefficient is a measure of similarity between two non-zero vectors
        that measures the cosine of the angle between them. For complex vectors,
        this function uses the Hermitian inner product.

        Args:
            x (np.ndarray): First complex vector.
            y (np.ndarray): Second complex vector.

        Returns:
            complex: The correlation coefficient between x and y. The result is a complex
                    number where the magnitude represents the similarity (1 being
                    identical, 0 being orthogonal) and the angle represents the
                    phase difference.

        Raises:
            ValueError: If the input vectors have different lengths or if either
                        vector has zero magnitude.

        Note:
            This function assumes that the input vectors are 1-dimensional arrays.
        """
        if x.shape != y.shape:
            raise ValueError("Input vectors must have the same shape")

        numerator = np.vdot(x, y)
        denominator = np.linalg.norm(x) * np.linalg.norm(y)

        if denominator == 0:
            raise ValueError("Correlation coefficient is undefined for zero vectors")

        return numerator / denominator
