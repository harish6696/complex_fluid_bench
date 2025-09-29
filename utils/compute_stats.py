"""
HDF5 Dataset Statistics Computation Utilities.

This module provides comprehensive statistical analysis capabilities for HDF5 datasets
commonly used in scientific computing and machine learning applications. It computes
essential statistics (mean, standard deviation, min, max, median, IQR) across multiple
files and groups, with support for temporal residual analysis and flexible data filtering.

Key Features:
- Multi-file statistical aggregation.
- Temporal residual statistics for time-series data
- Flexible filtering by groups, frames, and temporal stride
- Support for both on-the-fly and batch statistical computation
- Automatic channel expansion for multi-component fields
- Numerical precision(float64 internally)

Main Functions:
    compute_statistics: Primary API for computing dataset statistics
    
Internal Classes:
    _StatsAggregator: Utility class for accumulating statistics across data chunks

Statistical Measures:
    For each channel, the following statistics are computed:
    - mean: Arithmetic mean across all selected data points
    - std: Standard deviation (combined across groups using RMS)
    - min: Global minimum value
    - max: Global maximum value  
    - median: Median value (only when on_fly_stats=False)
    - iqr: Interquartile range (Q75 - Q25, only when on_fly_stats=False)



Memory Considerations:
    The module offers two computation modes:
    - on_fly_stats=True: Memory-efficient but cannot compute median and IQR
    - on_fly_stats=False: Higher memory usage but complete statistics
"""

import h5py
import numpy as np
from typing import List, Dict, Tuple
import re  # Added for parameter parsing


__all__ = [
    "compute_parameter_statistics",
]

def _parse_numeric_token(token: str) -> float:
    """Extract a floating point number from an arbitrary string token.

    The token often contains a parameter name followed by its value, e.g.
    "Re1000" → 1000.0 or "M0.6" → 0.6.  This helper searches the first
    numeric substring (including optional sign and exponent) and converts it
    to *float*.  If no numeric substring is found a *ValueError* is raised.
    """
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", token)
    if match:
        return float(match.group(0))
    raise ValueError(f"Cannot parse numeric value from token '{token}'.")


def compute_parameter_statistics(
    h5_paths: List[str],
    delimiter: str = "_",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """Compute min-max normalised simulation parameters encoded in group names.

    The parameters are encoded directly in the HDF5 group names using a delimiter (default: ``_``).
    For example::

        "Re100_M0.6"          -> [100.0, 0.6]
        "Re200_M0.3"          -> [200.0, 0.3]

    This utility scans *all* supplied files, extracts the numeric values of
    every parameter for every group, aggregates the full data set and finally
    applies min-max normalisation **independently for each parameter**::

        x_norm = (x - x_min) / (x_max - x_min + eps)

    Parameters
    ----------
    h5_paths : List[str]
        One or more HDF5 file paths to inspect.
    delimiter : str, default "_"
        Character used to split group names into parameter tokens.
    eps : float, default 1e-12
        Small constant preventing division by zero when *x_max == x_min*.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, Dict[str, float]]]
        1. ``normalized_params`` – A 2-D ``np.ndarray`` with shape
           *(N_groups, N_parameters)* containing the normalised parameters for
           every discovered group **in the order encountered**.
        2. ``param_ranges`` – A dictionary mapping the *parameter index* (int)
           to a ``{"min": float, "max": float}`` dictionary describing the
           range that was used for normalisation.

    Notes
    -----
    * If a token cannot be parsed into a float, a ``ValueError`` is raised.
    * All numeric conversions are executed in *float64* precision.
    * The function does **not** attempt to deduplicate groups across multiple
      files.  If the same group name occurs in two files, its parameter values
      will appear twice in the returned array.
    """
    if len(h5_paths) == 0:
        raise ValueError("h5_paths list is empty.")

    raw_params: List[List[float]] = []  # per-group list of parameter values

    # ------------------------------------------------------------------
    # Collect parameters from every group in every provided file
    # ------------------------------------------------------------------
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            for grp_name in f:
                tokens = grp_name.split(delimiter)
                values: List[float] = []
                for tok in tokens:
                    values.append(_parse_numeric_token(tok))
                raw_params.append(values)

    if len(raw_params) == 0:
        raise RuntimeError("No groups found in the supplied HDF5 files.")

    # Ensure parameter dimensionality is consistent
    n_params = len(raw_params[0])
    if not all(len(p) == n_params for p in raw_params):
        raise ValueError(
            "Inconsistent number of parameters detected across group names."
        )

    raw_arr = np.asarray(raw_params, dtype=np.float64)  # (N, P)

    # ------------------------------------------------------------------
    # Min-max normalisation per parameter dimension
    # ------------------------------------------------------------------
    mins = raw_arr.min(axis=0)
    maxs = raw_arr.max(axis=0)

    # denom = maxs - mins
    # denom[denom == 0.0] = eps  # avoid divide-by-zero for constant parameters

    # norm_arr = (raw_arr - mins) / denom

    # Assemble ranges into a dictionary for convenience
    param_ranges: Dict[int, Dict[str, float]] = {
        idx: {"min": float(mins[idx]), "max": float(maxs[idx])} for idx in range(n_params)
    }

    return param_ranges

# -----------------------------------------------------------------------------
# Re-normalization helper used during plotting
# -----------------------------------------------------------------------------

def re_normalize_data(arr: np.ndarray, stats: Dict[str, float], norm_strategy: str) -> np.ndarray:
    """Reverts normalization applied during loading for a single channel.
    Mainly used for plotting the normalized data.
    
    Parameters
    ----------
    arr : np.ndarray
        The normalized data to revert (any shape).
    stats : Dict[str, float]
        Dictionary containing either {"mean", "std"} or {"min", "max"} or {"median", "iqr"}. 
    norm_strategy : str
        Either "z_normalization" or "min_max_normalization" or "robust_normalization" or "no_normalization".
    
    Returns
    -------
    np.ndarray
        Renormalized array (new copy).
    """
    if norm_strategy == "z_normalization":
        return arr * stats["std"] + stats["mean"]
    elif norm_strategy == "min_max_normalization":
        return arr * (stats["max"] - stats["min"]) + stats["min"]
    elif norm_strategy == "robust_normalization":
        return arr * stats["iqr"] + stats["median"]
    elif norm_strategy == "no_normalization":
        return arr
    else:
        raise ValueError(f"Unknown normalization strategy: {norm_strategy}")

# -----------------------------------------------------------------------------
# Normalization helper used during data loading
# -----------------------------------------------------------------------------

def normalize_data(arr: np.ndarray, stats: Dict[str, float], strategy: str) -> np.ndarray:
    """Apply per-channel normalization to a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        The data to normalize (any shape).
    stats : Dict[str, float]
        Dictionary containing either {"mean", "std"} or {"min", "max"} or {"median", "iqr"}.
    strategy : str
        Either "z_normalization" or "min_max_normalization" or "robust_normalization" or "no_normalization".

    Returns
    -------
    np.ndarray
        Normalized array (new copy).
    """
    eps = 1e-12  # small constant to prevent divide-by-zero

    if strategy == "z_normalization":
        return (arr - stats["mean"]) / (stats["std"]  + eps)
    elif strategy == "min_max_normalization":
        return (arr - stats["min"]) / ((stats["max"] - stats["min"]) + eps)
    elif strategy == "robust_normalization":
        return (arr - stats["median"]) / (stats["iqr"] + eps)
    elif strategy == "no_normalization":
        return arr
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")


