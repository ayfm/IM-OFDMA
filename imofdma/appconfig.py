from typing import Dict, Any, List, Optional
import numpy as np
import yaml
from dataclasses import dataclass, field, asdict

from . import env


@dataclass
class BaseConfig:
    def __str__(self) -> str:
        return ", ".join(f"[{k}={v}]" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationConfig(BaseConfig):
    x_transmit_idx: int
    grouping_options: Optional[List[int]]
    checkpoint_percent: int
    random_seed: int
    batch_size: int
    iter_power: int
    iter_coeff: int
    n_iterations: int = field(init=False)

    def __post_init__(self):
        self.n_iterations = self.iter_coeff * (10**self.iter_power)

    def __str__(self) -> str:
        return (
            ">>> Simulation Parameters:\n"
            f"  > X_transmit      : {self.x_transmit_idx}\n"
            f"  > Grouping Option : {self.grouping_options}\n"
            f"  > Checkpoint      : {self.checkpoint_percent}%\n"
            f"  > Random Seed     : {self.random_seed}\n"
            f"  > Batch Size      : {self.batch_size}\n"
            f"  > Iter Power      : {self.iter_power}\n"
            f"  > Iter Coeff      : {self.iter_coeff}\n"
            f"  > N Iterations    : {self.n_iterations} (derived)\n"
        )


@dataclass
class IMOFDMAConfig(BaseConfig):
    M: int  # modulation order
    K: int  # total number of chunks
    Lk: int  # number of subcarriers in a chunk
    ## TODO: Implement support for multiple subcarriers per user
    ##La: int  # number of subcarriers allocated to each user
    Nk: int  # number of users to be served in a chunk
    SNRdB: float  # snr value in dB
    T: int  # channel coherence time
    var_H: float  # variance of channel coefficient distribution

    n_total_subcarriers: int = field(init=False)
    n_total_users: int = field(init=False)
    snr: float = field(init=False)
    var_W: float = field(init=False)
    nu_M: int = field(init=False)
    nu_L: int = field(init=False)
    nu_N: int = field(init=False)
    nu_K: int = field(init=False)
    n_lookup: int = field(init=False)

    def __post_init__(self):
        self.n_total_subcarriers = self.Lk * self.K
        self.n_total_users = self.Nk * self.K
        self.snr = 10 ** (self.SNRdB / 10)
        self.var_W = 1 / self.snr
        self.nu_M = int(np.log2(self.M))
        self.nu_L = int(np.log2(self.Lk))
        self.nu_N = self.nu_M + self.nu_L
        self.nu_K = self.nu_N * self.Nk
        self.n_lookup = 2**self.nu_K

    def __str__(self) -> str:
        return (
            ">>> IM-OFDMA Parameters:\n"
            f"  > M  : {self.M} - Modulation order\n"
            f"  > K  : {self.K} - Number of chunks\n"
            f"  > Lk : {self.Lk} - Number of subcarriers in a chunk\n"
            f"  > Lt : {self.n_total_subcarriers} - Total number of subcarriers (derived)\n"
            # f"  > La : {self.La} - Number of subcarriers allocated to each user\n"
            f"  > Nk : {self.Nk} - Number of user to be served in a chunk\n"
            f"  > Nt : {self.n_total_users} - Total number of users to be served (derived)\n"
            f"  > SNRdB : {self.SNRdB}\n"
            f"  > SNR   : {self.snr:.5f} (derived)\n"
            f"  > var_W : {self.var_W:.5f} - Variance of the internal noise (derived)\n"
            f"  > T     : {self.T} - Channel coherence time\n"
            f"  > var_H : {self.var_H:.5f} - Variance of the channel coefficient distribution\n"
            f"  > nu_M   : {self.nu_M} (derived)\n"
            f"  > nu_L   : {self.nu_L} (derived)\n"
            f"  > nu_N   : {self.nu_N} - Number of bits to be transmitted per user (derived)\n"
            f"  > nu_K   : {self.nu_K} - Number of bits to be transmitted per chunk (derived)\n"
            f"  > n_Xtable : {self.n_lookup} - Size of lookup table (derived)\n"
        )


@dataclass
class AppConfig(BaseConfig):
    sim: SimulationConfig
    im_ofdma: IMOFDMAConfig
    path_simulation: str = env.PATH_OUTPUTS

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "AppConfig":
        return cls(
            sim=SimulationConfig(**cfg_dict["simulation"]),
            im_ofdma=IMOFDMAConfig(**cfg_dict["im-ofdma"]),
        )

    def __str__(self) -> str:
        return f"{self.im_ofdma}\n{self.sim}"


def load_config(config_path: str) -> AppConfig:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return AppConfig.from_dict(cfg_dict)
