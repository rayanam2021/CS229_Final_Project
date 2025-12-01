import numpy as np
from roe.propagation import ROEDynamics

def apply_impulsive_dv(state_roe: np.ndarray, 
                       dv_rtn: np.ndarray, 
                       a_chief: float, 
                       n: float, 
                       tspan: np.ndarray,
                       e: float = 0.001,
                       i: float = 1.71,
                       omega: float = 0.0) -> np.ndarray:
    """
    Calculates the new ROE state after applying an impulsive Δv.

    Uses the control matrix derived from Gauss Variational Equations 
    (via ROEDynamics class) to map RTN velocity changes to ROE changes.

    Parameters
    ----------
    state_roe : np.ndarray
        Current ROE state vector [da, dlambda, dex, dey, dix, diy].
    dv_rtn : np.ndarray
        Impulsive velocity change [Δv_r, Δv_t, Δv_n] in m/s.
    a_chief : float
        Chief semi-major axis in km.
    n : float
        Chief mean motion in rad/s.
    tspan : np.ndarray
        Array containing the absolute time of the maneuver at index 0.
        Used to calculate True Anomaly (f).
    e : float
        Eccentricity (default 0.001 near-circular)
    i : float
        Inclination in radians (default ~98 deg)
    omega : float
        Argument of perigee in radians (default 0.0)

    Returns
    -------
    np.ndarray
        New ROE state vector.
    """
    
    # 1. Initialize Dynamics Model with full orbital parameters
    model = ROEDynamics(a_chief, e, i, omega)
    
    # 2. Determine True Anomaly (f) at time of burn
    # Assuming M ~ f for near-circular context if propagation is purely mean-anomaly based
    time_of_burn = tspan[0]
    f_burn = n * time_of_burn
    
    # 3. Calculate Change using Control Matrix
    # Convert m/s to km/s because 'a_chief' is in km and 'n' is rad/s
    dv_km_s = dv_rtn / 1000.0
    
    # Get Gamma (Control Matrix) from the model
    # The matrix B returned by model is scaled by 1/(na).
    Gamma = model.get_control_matrix(f_burn)
    
    delta_roe = Gamma @ dv_km_s
    
    # 4. Apply Change
    new_roe = state_roe + delta_roe
    
    return new_roe