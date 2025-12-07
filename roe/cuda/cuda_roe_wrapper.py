"""
Python wrapper for CUDA ROE propagation kernel using ctypes.
"""

import numpy as np
import ctypes
import os

# Load shared library
_lib_path = os.path.join(os.path.dirname(__file__), 'libroe_propagation.so')

try:
    _roe_lib = ctypes.CDLL(_lib_path)
    CUDA_AVAILABLE = True
except OSError:
    CUDA_AVAILABLE = False
    _roe_lib = None

if CUDA_AVAILABLE:
    # Define function signature
    _roe_lib.batch_propagate_roe_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # roe_initial_host [6]
        ctypes.POINTER(ctypes.c_double),  # dv_actions_host [num_actions x 3]
        ctypes.POINTER(ctypes.c_double),  # roe_final_host [num_actions x 6]
        ctypes.POINTER(ctypes.c_double),  # positions_rtn_host [num_actions x 3]
        ctypes.c_int,                     # num_actions
        ctypes.c_double,                  # a_chief
        ctypes.c_double,                  # e_chief
        ctypes.c_double,                  # i_chief
        ctypes.c_double,                  # omega
        ctypes.c_double,                  # n
        ctypes.c_double,                  # t_burn
        ctypes.c_double,                  # dt
    ]
    _roe_lib.batch_propagate_roe_cuda.restype = None


def batch_propagate_roe_cuda(
    roe_initial: np.ndarray,
    dv_actions: np.ndarray,
    a_chief: float,
    e_chief: float,
    i_chief: float,
    omega: float,
    n: float,
    t_burn: float,
    dt: float
):
    """
    Propagate multiple ROE states with different delta-v maneuvers using CUDA.

    Parameters
    ----------
    roe_initial : np.ndarray
        Initial ROE state [6]: [da, dlambda, dex, dey, dix, diy]
    dv_actions : np.ndarray
        Array of delta-v actions [num_actions x 3] in m/s
    a_chief : float
        Chief semi-major axis [km]
    e_chief : float
        Eccentricity
    i_chief : float
        Inclination [rad]
    omega : float
        Argument of perigee [rad]
    n : float
        Mean motion [rad/s]
    t_burn : float
        Burn time [s]
    dt : float
        Propagation time step [s]

    Returns
    -------
    roe_final : np.ndarray
        Final ROE states [num_actions x 6]
    positions_rtn : np.ndarray
        RTN positions [num_actions x 3] in km
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            f"CUDA ROE propagation library not found at {_lib_path}. "
            "Compile with: nvcc -shared -o libroe_propagation.so roe_propagation_kernel.cu"
        )

    # Convert to float64 (double precision for accuracy)
    roe_initial = np.asarray(roe_initial, dtype=np.float64)
    dv_actions = np.asarray(dv_actions, dtype=np.float64)

    if roe_initial.shape != (6,):
        raise ValueError(f"roe_initial must have shape (6,), got {roe_initial.shape}")

    if dv_actions.ndim != 2 or dv_actions.shape[1] != 3:
        raise ValueError(f"dv_actions must have shape (num_actions, 3), got {dv_actions.shape}")

    num_actions = dv_actions.shape[0]

    # Allocate output arrays
    roe_final = np.zeros((num_actions, 6), dtype=np.float64)
    positions_rtn = np.zeros((num_actions, 3), dtype=np.float64)

    # Ensure contiguous arrays
    roe_initial = np.ascontiguousarray(roe_initial)
    dv_actions = np.ascontiguousarray(dv_actions)

    # Call CUDA kernel
    _roe_lib.batch_propagate_roe_cuda(
        roe_initial.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dv_actions.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        roe_final.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        positions_rtn.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(num_actions),
        ctypes.c_double(a_chief),
        ctypes.c_double(e_chief),
        ctypes.c_double(i_chief),
        ctypes.c_double(omega),
        ctypes.c_double(n),
        ctypes.c_double(t_burn),
        ctypes.c_double(dt)
    )

    return roe_final, positions_rtn


def is_cuda_available():
    """Check if CUDA ROE propagation is available."""
    return CUDA_AVAILABLE
