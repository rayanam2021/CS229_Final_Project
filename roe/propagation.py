import numpy as np

def propagateGeomROE(deltaROE, a, e, i, omega, n, tspan):
    """
    Propagates relative orbital elements over time using the geometric ROE model.

    Parameters:
        deltaROE : array-like, shape (6,)
            [da, dlambda, dex, dey, dix, diy]
        a, e, i, omega, n : floats
            Chief orbit parameters (semi-major axis [km], ecc, inc [rad], arg. perigee [rad], mean motion [rad/s])
        tspan : array
            Time array in seconds

    Returns:
        t : ndarray
        rho : ndarray, shape (3, len(tspan)) — relative position in RTN frame (km)
        rhodot : ndarray, shape (3, len(tspan)) — relative velocity in RTN frame (km/s)
    """
    delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy = deltaROE

    # Mean argument of latitude
    M = n * tspan

    # Relative position and velocity in RTN (from D'Amico 2010)
    rho_r = a * (delta_ex * np.sin(M) - delta_ey * np.cos(M))
    rho_t = a * (delta_lambda + 2 * delta_ex * np.cos(M) + 2 * delta_ey * np.sin(M))
    rho_n = a * (delta_ix * np.sin(M) - delta_iy * np.cos(M))

    rhodot_r = a * n * (delta_ex * np.cos(M) + delta_ey * np.sin(M))
    rhodot_t = a * n * (-2 * delta_ex * np.sin(M) + 2 * delta_ey * np.cos(M))
    rhodot_n = a * n * (delta_ix * np.cos(M) + delta_iy * np.sin(M))

    rho = np.vstack((rho_r, rho_t, rho_n))
    rhodot = np.vstack((rhodot_r, rhodot_t, rhodot_n))

    return rho, rhodot


def rtn_to_roe(rho, rhodot, a, n, tspan):
    """
    Converts relative position/velocity in RTN frame to relative orbital elements (ROEs)
    using the mean argument of latitude M.

    Parameters:
        rho : array-like, shape (3,)
            [rho_r, rho_t, rho_n] relative position in RTN [km]
        rhodot : array-like, shape (3,)
            [rhodot_r, rhodot_t, rhodot_n] relative velocity in RTN [km/s]
        a : float
            Chief semi-major axis [km]
        n : float
            Chief mean motion [rad/s]
        M : float
            Mean argument of latitude (n * t) [rad]

    Returns:
        deltaROE : ndarray, shape (6,)
            [delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy]
    """
    rho_r, rho_t, rho_n = rho
    rhodot_r, rhodot_t, rhodot_n = rhodot
    M = n * tspan[0]

    sin_M, cos_M = np.sin(M), np.cos(M)

    # δe_x, δe_y from radial components
    delta_ex = (1/a) * (rho_r * sin_M + (rhodot_r/n) * cos_M)
    delta_ey = (1/a) * (-rho_r * cos_M + (rhodot_r/n) * sin_M)

    # δi_x, δi_y from normal components
    delta_ix = (1/a) * (rho_n * sin_M + (rhodot_n/n) * cos_M)
    delta_iy = (1/a) * (-rho_n * cos_M + (rhodot_n/n) * sin_M)

    # δλ from along-track component
    delta_lambda = (rho_t / a) - 2 * (delta_ex * cos_M + delta_ey * sin_M)

    delta_a = 0.0  # not included in linear geometric model

    return [delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy]


def rotation_rtn_to_eci(i, omega, M):
    """Returns the rotation matrix from RTN to ECI for given chief orbital elements."""
    # Simplified: assume argument of latitude u = omega + M
    u = omega + M
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_i, sin_i = np.cos(i), np.sin(i)

    # Rotation matrix from RTN to ECI
    R = np.array([
        [cos_u, -sin_u * cos_i, sin_u * sin_i],
        [sin_u, cos_u * cos_i, -cos_u * sin_i],
        [0, sin_i, cos_i]
    ])
    return R


def get_relative_positions_cartesian(deltaROE, a, e, i, omega, n, tspan):
    """
    Propagates relative orbital elements and returns chaser positions in ECI frame.
    """
    t, rho_rtn, _ = propagateGeomROE(deltaROE, a, e, i, omega, n, tspan)

    positions_eci = []
    for idx, M in enumerate(n * tspan):
        R_rtn2eci = rotation_rtn_to_eci(i, omega, M)
        pos_eci = R_rtn2eci @ rho_rtn[:, idx]
        positions_eci.append(pos_eci)

    return np.array(positions_eci)