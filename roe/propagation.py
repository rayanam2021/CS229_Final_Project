import numpy as np

class ROEDynamics:
    """
    Class to handle Relative Orbital Elements (ROE) dynamics based on 
    Willis's Dissertation "Analytical Theory of Satellite Relative Motion" (2023).
    
    ROE Definition (Quasi-Nonsingular):
    roe = [da, dlambda, dex, dey, dix, diy]
    """
    
    def __init__(self, a, e, i, omega, mu=398600.4418):
        self.a = a
        self.e = e
        self.i = i
        self.omega = omega
        self.mu = mu
        self.n = np.sqrt(mu / a**3) # Mean motion
        self.eta = np.sqrt(1 - e**2) # Eccentricity factor

    def get_stm(self, dt):
        """State Transition Matrix (STM) Phi(t, t0)."""
        Phi = np.eye(6)
        # Linear drift of relative mean longitude (Willis Eq 2.46) [cite: 1077]
        Phi[1, 0] = -1.5 * self.n * dt
        return Phi

    def propagate(self, roe_initial, dt, second_order=True):
        """Propagates ROE state vector forward by duration dt."""
        roe_0 = np.array(roe_initial)
        
        # 1. Linear Propagation
        Phi = self.get_stm(dt)
        roe_next = Phi @ roe_0
        
        # 2. Second-Order Correction (Willis Eq 3.41) [cite: 2181]
        if second_order:
            da = roe_0[0]
            second_order_term = (15.0 / 8.0) * self.n * dt * (da**2)
            roe_next[1] += second_order_term
            
        return roe_next

    def get_control_matrix(self, f):
        """
        Calculates Control Matrix (B) mapping Delta-V_RTN to Delta-ROE.
        Ensures consistency with the map_roe_to_rtn function to avoid position jumps.
        Derived from Gauss Variational Equations.
        """
        theta = self.omega + f # Argument of latitude u
        
        # Precompute trig
        st = np.sin(theta)
        ct = np.cos(theta)
        sf = np.sin(f) 
        
        B = np.zeros((6, 3))
        # Normalization factor: GVEs usually have 1/na or 2/na. 
        # We factor out 1/na here.
        inv_na = 1.0 / (self.n * self.a) 
        
        # --- 1. Relative Semi-Major Axis (da) ---
        # da = 2/n * vt -> B[0,1] = 2
        B[0, 0] = 0.0 
        B[0, 1] = 2.0
        
        # --- 2. Relative Mean Longitude (dlambda) ---
        # dlambda = -2/n*a * vr
        B[1, 0] = -2.0
        B[1, 1] = -3.0 * self.e * sf # Higher order term kept for drifting logic
        
        # --- 3. Relative Eccentricity Vector (dex, dey) ---
        # Uses consistent approximation D(dex) ~ sin(u)*dvr + 2cos(u)*dvt
        B[2, 0] = st        # sin(u)
        B[2, 1] = 2.0 * ct  # 2 cos(u)
        
        B[3, 0] = -ct       # -cos(u)
        B[3, 1] = 2.0 * st  # 2 sin(u)

        # --- 4. Relative Inclination Vector (dix, diy) ---
        # Uses consistent approximation D(dix) ~ cos(u)*dvn
        B[4, 2] = ct # cos(u)
        B[5, 2] = st # sin(u)

        return B * inv_na

def map_roe_to_rtn(roe, a, n, f=0.0, omega=0.0):
    """
    Geometric mapping from ROE to RTN Position/Velocity.
    Standard Linear Mapping (Eq 2.45 in Willis 2023) [cite: 1051]
    """
    da, dl, dex, dey, dix, diy = roe
    u = omega + f 
    
    su = np.sin(u)
    cu = np.cos(u)
    
    # Relative Position (Linear mapping)
    r_r = -a * cu * dex - a * su * dey + a * da
    r_t = a * dl + 2 * a * su * dex - 2 * a * cu * dey
    r_n = a * su * dix - a * cu * diy
    
    # Relative Velocity
    v_r = a * n * (su * dex - cu * dey)
    v_t = 2 * a * n * (cu * dex + su * dey) - 1.5 * n * a * da 
    v_n = a * n * (cu * dix + su * diy)
    
    return np.array([r_r, r_t, r_n]), np.array([v_r, v_t, v_n])

def propagateGeomROE(deltaROE, a, e, i, omega, n, tspan, t0=0.0):
    """
    Propagates ROEs and maps to RTN.
    """
    model = ROEDynamics(a, e, i, omega)
    rho_list = []
    rhodot_list = []
    times = np.atleast_1d(tspan)
    
    for t_abs in times:
        dt = t_abs - t0
        roe_t = model.propagate(deltaROE, dt, second_order=True)
        M_t = n * t_abs
        r_vec, v_vec = map_roe_to_rtn(roe_t, a, n, f=M_t, omega=omega)
        rho_list.append(r_vec)
        rhodot_list.append(v_vec)
        
    return np.array(rho_list).T, np.array(rhodot_list).T

def rtn_to_roe(rho, rhodot, a, n, tspan):
    """Converts RTN to ROE. tspan[0] must be absolute time."""
    rho_r, rho_t, rho_n = rho
    rhodot_r, rhodot_t, rhodot_n = rhodot
    
    M = n * tspan[0]
    u = 0.0 + M 
    su = np.sin(u)
    cu = np.cos(u)

    delta_ix = (rho_n * su + (rhodot_n/n) * cu) / a
    delta_iy = (-rho_n * cu + (rhodot_n/n) * su) / a

    delta_a = (2.0 * rhodot_t) / (n * a) + (3.0 * rho_r) / a 
    
    delta_ex = (1.0/a) * (3.0*rho_r*su + 2.0*(rhodot_t/n)*su + (rhodot_r/n)*cu)
    delta_ey = (1.0/a) * (3.0*rho_r*cu + 2.0*(rhodot_t/n)*cu - (rhodot_r/n)*su)
    
    delta_lambda = (rho_t / a) - 2.0 * (delta_ex * cu + delta_ey * su)

    return [delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy]