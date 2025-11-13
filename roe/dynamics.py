import numpy as np

def apply_impulsive_dv(state_roe: np.ndarray, 
                       dv_rtn: np.ndarray, 
                       a_chief: float, 
                       n: float, 
                       tspan: np.ndarray) -> np.ndarray:
    """
    Calculates the new ROE state after applying an impulsive Δv.

    This function uses Gauss's Variational Equations (GVEs) adapted for
    Relative Orbital Elements (ROEs) on a near-circular orbit. It maps an
    instantaneous change in velocity (Δv) in the RTN frame to an
    instantaneous change in the ROE state vector.

    Parameters
    ----------
    state_roe : np.ndarray
        The current (old) ROE state vector [da, dlambda, dex, dey, dix, diy].
    dv_rtn : np.ndarray
        The impulsive velocity change to apply [Δv_r, Δv_t, Δv_n] in m/s.
    a_chief : float
        Chief semi-major axis in km.
    n : float
        Chief mean motion in rad/s.
    tspan : np.ndarray
        The propagation time in seconds. Used to calculate the
        mean anomaly (M) at the time of the impulse.

    Returns
    -------
    np.ndarray
        The new ROE state vector after the impulse.
    """
    
    # --- 1. Unpack Inputs ---
    da_old, dl_old, dex_old, dey_old, dix_old, diy_old = state_roe
    
    dv_rtn_kms = dv_rtn
    dv_r, dv_t, dv_n = dv_rtn_kms

    # --- 2. Calculate Mean Anomaly (M) ---
    # We assume M0=0 at t=0. 'M' is the orbital position where
    # the impulse is applied.
    M = n * tspan[0]
    cos_M = np.cos(M)
    sin_M = np.sin(M)

    # --- 3. Calculate Instantaneous Change in ROEs (via GVEs) ---
    # These equations map [dv_r, dv_t, dv_n] to [Δda, Δdlambda, ...]
    
    # Change in relative semi-major axis (da)
    delta_da = (2.0 / n) * dv_t

    # Change in relative mean longitude (dlambda)
    delta_dl = (-2.0 / (n * a_chief)) * dv_r

    # Change in relative eccentricity vector (dex, dey)
    delta_dex = (sin_M / (n * a_chief)) * dv_r + (2.0 * cos_M / (n * a_chief)) * dv_t
    delta_dey = (-cos_M / (n * a_chief)) * dv_r + (2.0 * sin_M / (n * a_chief)) * dv_t
    
    # Change in relative inclination vector (dix, diy)
    delta_dix = (cos_M / (n * a_chief)) * dv_n
    delta_diy = (sin_M / (n * a_chief)) * dv_n
    
    # --- 4. Calculate New ROE State ---
    # The new state is the old state plus the instantaneous change
    da_new    = da_old    + delta_da
    dl_new    = dl_old    + delta_dl
    dex_new   = dex_old   + delta_dex
    dey_new   = dey_old   + delta_dey
    dix_new   = dix_old   + delta_dix
    diy_new   = diy_old   + delta_diy

    return [da_new, dl_new, dex_new, dey_new, dix_new, diy_new]