import os
import numpy as np

# Directory Constants
DIR_PACKAGE: str = os.path.dirname(os.path.realpath(__file__))
PATH_CONFIG: str = os.path.join(DIR_PACKAGE, "params.yaml")
PATH_OUTPUTS: str = os.path.join(DIR_PACKAGE, "..", "simulations")

# Simulation Constants
PRECISION: int = 5
RND: np.random.RandomState = np.random.RandomState(None)


def ensure_output_directory_exists() -> None:
    """Create the output directory if it doesn't exist."""
    os.makedirs(PATH_OUTPUTS, exist_ok=True)


# Ensure output directory exists when this module is imported
ensure_output_directory_exists()
