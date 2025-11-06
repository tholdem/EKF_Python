import numpy as np
import cmsisdsp as dsp

DTYPE = np.float32

def compute_dvdot_dp(state, we, R_phi, R_lamb, dR_phi_dphi, dR_lamb_dphi, dg_dphi, dg_dh):
    """
    phi :Latitude
    h : Height
    vn, ve, vd :  Velocity
    we : Earth rotation 
    R_phi, R_lamb : Radii of curvature 
    dR_phi_dphi, dR_lamb_dphi : Derivatives of radii wrt latitude 
    dg_dphi, dg_dh :Derivatives of gravity

    """
    phi, h, vn, ve, vd = state[4], state[6], state[7], state[8], state[9]
    deg2rad = DTYPE(np.pi / 180.0)
    phi_rad = deg2rad * phi
    cos_phi = dsp.arm_cos_f32(phi_rad)
    sin_phi = dsp.arm_sin_f32(phi_rad)
    sec_phi = 1.0 / cos_phi
    tan_phi = sin_phi / cos_phi

    # Denominators
    denom_Rphi = (R_phi + h)
    denom_Rlam = (R_lamb + h)

    # Eqn 7.82a
    Y11 = -(ve**2 * sec_phi**2) / denom_Rlam \
          + (ve**2 * tan_phi) / (denom_Rlam**2) * dR_lamb_dphi \
          - 2 * we * ve * cos_phi \
          - (vn * vd) / (denom_Rphi**2) * dR_phi_dphi

    # Eqn 7.82b
    Y13 = (ve**2 * tan_phi) / (denom_Rlam**2) \
          - (vn * vd) / (denom_Rphi**2)

    # Eqn 7.82c
    Y21 = (ve * vn * sec_phi**2) / denom_Rlam \
          - (ve * vn * tan_phi) / (denom_Rlam**2) * dR_lamb_dphi \
          + 2 * we * vn * cos_phi \
          - (ve * vd) / (denom_Rlam**2) * dR_lamb_dphi \
          - 2 * we * vd * sin_phi

    # Eqn 7.82d
    Y23 = -ve * ((vn * tan_phi + vd) / (denom_Rlam**2))

    # Eqn 7.82e
    Y31 = (ve**2) / (denom_Rlam**2) * dR_lamb_dphi \
          + (vn**2) / (denom_Rphi**2) * dR_phi_dphi \
          + 2 * we * ve * sin_phi \
          + dg_dphi

    # Eqn 7.82f
    Y33 = (ve**2) / (denom_Rlam**2) \
          + (vn**2) / (denom_Rphi**2) \
          + dg_dh

    # Assemble final matrix
    dvdot_dp = np.array([
        [Y11, 0.0, Y13],
        [Y21, 0.0, Y23],
        [Y31, 0.0, Y33]
    ], dtype=DTYPE)
    np.savetxt("dVdPdot.csv", dvdot_dp, delimiter=",", fmt="%.9e")

    return dvdot_dp

def compute_dvdot_dv(state, we, R_phi, R_lamb):
    """
    Parameters
    ----------
    phi : Latitude 
    h : Height 
    vn, ve, vd : Velocity 
    we :Earth rotation 
    R_phi, R_lamb :  Radii of curvature in meridian and prime vertical [m]

    """
    phi, h, vn, ve, vd = state[4], state[6], state[7], state[8], state[9]
    # CMSIS-DSP trigs
    deg2rad = DTYPE(np.pi / 180.0)
    phi_rad = deg2rad * phi
    cos_phi = dsp.arm_cos_f32(phi_rad)
    sin_phi = dsp.arm_sin_f32(phi_rad)
    tan_phi = sin_phi / cos_phi

    denom_Rphi = (R_phi + h)
    denom_Rlam = (R_lamb + h)

    # Eqn 7.83a
    Z11 = vd / denom_Rphi
    Z12 = (-2 * ve * tan_phi) / denom_Rlam + 2 * we * sin_phi
    Z13 = vn / denom_Rphi

    # Eqn 7.83b
    Z21 = (ve * tan_phi) / denom_Rlam + 2 * we * sin_phi
    Z22 = (vd + vn * tan_phi) / denom_Rlam
    Z23 = ve / denom_Rlam + 2 * we * cos_phi

    # Eqn 7.83c
    Z31 = (-2 * vn) / denom_Rphi
    Z32 = (-2 * ve) / denom_Rlam - 2 * we * cos_phi

    # Assemble the final 3x3 matrix
    dvdot_dv = np.array([
        [Z11, Z12, Z13],
        [Z21, Z22, Z23],
        [Z31, Z32, 0.0]
    ], dtype=DTYPE)
    np.savetxt("dVdVdot.csv", dvdot_dv, delimiter=",", fmt="%.9e")

    return dvdot_dv


def compute_dpdot_dv(state, R_phi, R_lamb):
    phi, h = state[4], state[6]
    deg2rad = DTYPE(np.pi / 180.0)
    rad2deg = DTYPE(180.0 / np.pi)
    phi_rad = deg2rad * phi
    cos_phi = dsp.arm_cos_f32(phi_rad)
    sec_phi = 1.0 / cos_phi

    denom_Rphi = (R_phi + h)
    denom_Rlam = (R_lamb + h)

    m11 = rad2deg * 1.0 / denom_Rphi
    m22 = rad2deg *sec_phi / denom_Rlam
    m33 = -1.0

    dpdot_dv = np.diag([m11, m22, m33]).astype(DTYPE)
    np.savetxt("dPdVdot.csv", dpdot_dv, delimiter=",", fmt="%.9e")

    return dpdot_dv


def compute_dpdot_dp(state, R_phi, R_lamb, dR_phi_dphi, dR_lamb_dphi):
    """
    phi :Latitude
    h : Height
    vn, ve, vd :  Velocity
    we : Earth rotation 
    R_phi, R_lamb : Radii of curvature 
    dR_phi_dphi, dR_lamb_dphi : Derivatives of radii wrt latitude 
    dg_dphi, dg_dh :Derivatives of gravity

    """
    phi, h, vn, ve = state[4], state[6], state[7], state[8]
    deg2rad = DTYPE(np.pi / 180.0)
    rad2deg = DTYPE(180.0 / np.pi)
    phi_rad = deg2rad * phi
    cos_phi = dsp.arm_cos_f32(phi_rad)
    sin_phi = dsp.arm_sin_f32(phi_rad)
    sec_phi = 1.0 / cos_phi
    tan_phi = sin_phi / cos_phi

    # Denominators
    denom_Rphi = (R_phi + h)
    denom_Rlam = (R_lamb + h)

    

    # Eqn 7.80a terms
    m11 = -vn / (denom_Rphi ** 2) * dR_phi_dphi 
    m13 = rad2deg * -vn / (denom_Rphi ** 2) 

    m21 = -(ve * sec_phi) / (denom_Rlam ** 2) * dR_lamb_dphi \
          + (ve * sec_phi * tan_phi) / (denom_Rlam)
    m23 = rad2deg * -ve * sec_phi / (denom_Rlam ** 2)

    # Assemble final matrix
    dpdot_dp = np.array([
        [m11, 0.0, m13],
        [m21, 0.0, m23],
        [0.0,  0.0, 0.0]
    ], dtype=DTYPE)
    np.savetxt("dPdPdot.csv", dpdot_dp, delimiter=",", fmt="%.9e")

    return dpdot_dp


def compute_ahat_n(state, a_meas, DCM_b2n):
    """
    Compute acceleration in NED frame - matches MATLAB compute_ahat()
    """
    q = state[0:4]
    accel_bias = state[13:16]
    accel_sf = state[19:22].astype(DTYPE)
    # Apply calibrations
    ones = np.ones(3, dtype=DTYPE)
    sf_correction = dsp.arm_add_f32(ones, accel_sf)
    # Compensate bias and scale factor
    a_meas = a_meas.astype(DTYPE)
    a_body = np.zeros(3, dtype=DTYPE)
    for i in range(3):
        a_body[i] = (a_meas[i] - accel_bias[i]) / sf_correction[i]
    # Transform to NED frame
    # Matrix-vector multiply using CMSIS
    a_ned = dsp.arm_mat_vec_mult_f32(DCM_b2n, a_body.astype(DTYPE))

    np.savetxt("AhatN.csv", a_ned, delimiter=",", fmt="%.9e")
    return a_ned


def compute_dwdv(state, R_phi, R_lamb):
    """

    Parameters
    ----------
    phi : float
        Latitude [rad]
    h : float
        Altitude [m]
    R_phi, R_lamb : float
        Radii of curvature [m]

    """
    phi, h = state[4], state[6]
    deg2rad = DTYPE(np.pi / 180.0)
    phi_rad = deg2rad * phi
    sin_phi = dsp.arm_sin_f32(phi_rad)
    cos_phi = dsp.arm_cos_f32(phi_rad)
    tan_phi = sin_phi / cos_phi
    # === Eqn (7.74b) components ===
    m12 = 1.0 / (R_lamb + h)
    m21 = -1.0 / (R_phi + h)
    m32 = -tan_phi / (R_lamb + h)

    # === Assemble Jacobian ===
    dwdv = np.array([
        [0.0, m12, 0.0],
        [m21, 0.0, 0.0],
        [0.0, m32, 0.0]
    ], dtype=np.float32)

    np.savetxt("dWdV.csv", dwdv, delimiter=",", fmt="%.9e")

    return dwdv

def compute_dwdp(state, we, R_phi, R_lamb,dR_phi_dphi, dR_lamb_dphi):
    """
    ----------
    phi : Latitude [rad]
    h : Altitude [m]
    vn, ve : North , east velocity 
    we : Earth rotation 
    R_phi, R_lamb : Radii of curvature
    dR_phi_dphi, dR_lamb_dphi : Derivative wrt latitude

    """
    phi, h, vn, ve = state[4], state[6], state[7], state[8]
    deg2rad = DTYPE(np.pi / 180.0)
    phi_rad = deg2rad * phi
    # === Trig functions (radians) ===

    cos_phi = dsp.arm_cos_f32(phi_rad)
    sin_phi = dsp.arm_sin_f32(phi_rad)
    tan_phi = sin_phi / cos_phi
    sec_phi = 1.0 / cos_phi

    # === Precompute radius sums ===
    Rl_h = R_lamb + h
    Rp_h = R_phi + h

    # === Eqn (7.74a) terms ===
    m11 = -we * sin_phi - ve / (Rl_h ** 2) * dR_lamb_dphi
    m13 = -ve / (Rl_h ** 2)
    m21 = vn / (Rp_h ** 2) * dR_phi_dphi
    m23 = vn / (Rp_h ** 2)
    m31 = (-we * cos_phi
           - (ve * sec_phi ** 2) / Rl_h
           + (ve * tan_phi) / (Rl_h ** 2) * dR_lamb_dphi)
    m33 = (ve * tan_phi) / (Rl_h ** 2)

    # === Assemble Jacobian ===
    dwdp = np.array([
        [m11, 0.0, m13],
        [m21, 0.0, m23],
        [m31, 0.0, m33]
    ], dtype=np.float32)

    np.savetxt("dWdP.csv", dwdp, delimiter=",", fmt="%.9e")

    return dwdp

def skew_symmetric(v):
    """Build skew-symmetric matrix - matches MATLAB skew()"""
    v = v.astype(DTYPE)

    skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=DTYPE)

    return skew


def compute_G(state, DCM_b2n):
    sf_g = state[16:19]
    sf_a = state[19:22]
    G11 = -np.diag(1.0 / (1.0 + sf_g))
    G33 = -dsp.arm_mat_mult_f32(DCM_b2n, np.diag(1.0 / (1.0 + sf_a)))[1]
    # Build G matrix
    G = np.zeros((21, 12), dtype=DTYPE)

    # Attitude
    G[0:3, 0:3] = G11
    # Velocity
    G[6:9, 6:9] = G33
    G[9:12, 3:6] = np.eye(3, dtype=DTYPE)  # Gyro bias
    G[15:18, 9:12] = np.eye(3, dtype=DTYPE)  # Accel bias
    return G

def compute_F(state, a_meas, w_meas, R_phi, R_lamb, dR_phi_dphi, dR_lamb_dphi, dg_dphi, dg_dh, DCM_b2n, we):
    q, bias_g, sf_g, bias_a, sf_a = state[0:4], state[10:13], state[13:16], state[16:19], state[19:22]
    # ==== Rotation matrices ====
    DCM_n2b = dsp.arm_mat_trans_f32(DCM_b2n)[1]

    # ==== F11 ====
    F11_vec = -1.0 / (1.0 + sf_g) * (w_meas - bias_g) #elementwise product of vectors
    F11 = skew_symmetric(F11_vec)  # 3x3 numpy array

    # ==== F12, F13 ====
    dwdp = compute_dwdp(state, we, R_phi, R_lamb, dR_phi_dphi, dR_lamb_dphi)
    F12 = - dsp.arm_mat_mult_f32(DCM_n2b, dwdp)[1]

    dwdv = compute_dwdv(state, R_phi, R_lamb)
    F13 = - dsp.arm_mat_mult_f32(DCM_n2b, dwdv)[1]

    # ==== F14 ====
    F14 = np.diag(-1.0 / (1.0 + sf_g))

    # ==== F16 ====
    Omega = np.diag(w_meas)
    Bg = np.diag(bias_g)
    F16 = - dsp.arm_mat_sub_f32(Omega, Bg)[1]

    # ==== F22, F23 ====
    F22 = compute_dpdot_dp(state, R_phi, R_lamb, dR_phi_dphi, dR_lamb_dphi)
    F23 = compute_dpdot_dv(state, R_phi, R_lamb)

    # ==== F31 ====
    ahat_n = compute_ahat_n(state, a_meas, DCM_b2n)
    ahat_b = dsp.arm_mat_vec_mult_f32(DCM_n2b, ahat_n)
    F31 = -dsp.arm_mat_mult_f32(DCM_b2n, skew_symmetric(ahat_b))[1]

    # ==== F32, F33 ====
    F32 = compute_dvdot_dp(state, we, R_phi, R_lamb, dR_phi_dphi, dR_lamb_dphi, dg_dphi, dg_dh)
    F33 = compute_dvdot_dv(state, we, R_phi, R_lamb)
    
    # ==== F35 ====
    diag_sf_a = np.diag(1.0 / (1.0 + sf_a)).astype(DTYPE)
    F35 = -dsp.arm_mat_mult_f32(DCM_b2n, diag_sf_a)[1]

    # ==== F37 ====
    diag_diff = np.diag(a_meas - bias_a).astype(DTYPE)
    F37 = -dsp.arm_mat_mult_f32(DCM_b2n, diag_diff)[1]

    F = np.zeros((21, 21), dtype=DTYPE)

    # Attitude
    F[0:3, 0:3] = F11
    F[0:3, 3:6] = F12
    F[0:3, 6:9] = F13
    F[0:3, 9:12] = F14
    F[0:3, 15:18] = F16

    # Position
    F[3:6, 3:6] = F22
    F[3:6, 6:9] = F23

    # Velocity
    F[6:9, 0:3] = F31
    F[6:9, 3:6] = F32
    F[6:9, 6:9] = F33
    F[6:9, 12:15] = F35
    F[6:9, 18:21] = F37

    np.savetxt("computeF.csv", F, delimiter=",", fmt="%.9e")

    # The rest (bias/scale factor dynamics) remain zero
    return F

def compute_gravity(lat_rad, alt):
    """WGS84 gravity model and derivatives - matches MATLAB compute_g_dg()
    Returns:
        g, dg_dphi, dg_dh where phi is in radians and h is altitude in meters
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lat_sq = sin_lat * sin_lat
    sin_2lat = np.sin(2.0 * lat_rad)
    sin_2lat_sq = sin_2lat * sin_2lat
    sin_4lat = np.sin(4.0 * lat_rad)

    # Surface gravity
    g = 9.780327 * (1 + 5.3024e-3 * sin_lat_sq - 5.8e-6 * sin_2lat_sq) \
        - (3.0877e-6 - 4.4e-9 * sin_lat_sq) * alt + 7.2e-14 * alt ** 2

    # Derivative w.r.t latitude (radians)
    dg_dphi = 9.780327 * (5.3024e-3 * sin_2lat - 4.64e-5 * 0.25 * sin_4lat) \
              + 4.4e-9 * alt * sin_2lat

    # Derivative w.r.t altitude
    dg_dh = -3.0877e-6 + 4.4e-9 * sin_lat_sq + 1.44e-13 * alt


    return DTYPE(g), DTYPE(dg_dphi), DTYPE(dg_dh)

def compute_radii(lat_rad, a, e2):
    """Compute Earth radii and derivatives - matches MATLAB compute_radii()
    Returns:
        R_phi, R_lambda, dR_phi_dphi, dR_lambda_dphi (derivatives w.r.t latitude in radians)
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    sin_lat_sq = sin_lat * sin_lat
    f = 1.0 - e2 * sin_lat_sq
    sqrt_f = np.sqrt(f)

    # Radii of curvature
    R_phi = a * (1.0 - e2) / (sqrt_f**3)  # Eqn 7.69a (meridian)
    R_lambda = a / sqrt_f              # Eqn 7.69b (prime vertical)

    # Derivatives w.r.t phi (radians) - Eqns 7.75a, 7.75b
    dR_phi_dphi = 3.0 * a * (1.0 - e2) * e2 * sin_lat * cos_lat / sqrt_f**5
    dR_lambda_dphi = a * e2 * sin_lat * cos_lat / (sqrt_f**3)

    return DTYPE(R_phi), DTYPE(R_lambda), DTYPE(dR_phi_dphi), DTYPE(dR_lambda_dphi)

def compute_Pdot(state, P, Q, a_meas, w_meas, DCM_b2n, we, a, e2):
    #Get values from F feom state vector
    q = state[0:4]
    phi_rad = state[4] * np.pi / 180.0
    alt = state[6]
    R_phi, R_lamb, dR_phi_dphi, dR_lamb_dphi = compute_radii(phi_rad, a, e2)
    _, dg_dphi, dg_dh = compute_gravity(phi_rad, alt)
    F = compute_F(state, a_meas, w_meas, R_phi, R_lamb, dR_phi_dphi, dR_lamb_dphi, dg_dphi, dg_dh, DCM_b2n, we)
    G = compute_G(state, DCM_b2n)

    # CMSIS-DSP matrix operations
    # Pdot = F*P + P*F' + G*Q*G'
    FP = dsp.arm_mat_mult_f32(F, P)[1]
    Ft = dsp.arm_mat_trans_f32(F)[1]
    PFt = dsp.arm_mat_mult_f32(P, Ft)[1]
    GQ = dsp.arm_mat_mult_f32(G, Q)[1]
    Gt = dsp.arm_mat_trans_f32(G)[1]
    GQGt = dsp.arm_mat_mult_f32(GQ, Gt)[1]
    FP_plus_PFt = dsp.arm_mat_add_f32(FP, PFt)[1]
    Pdot = dsp.arm_mat_add_f32(FP_plus_PFt, GQGt)[1]

    return Pdot



# === Include your functions here ===
# (Paste your full definitions of compute_F, compute_G, compute_Pdot, etc. above this test block)
# ------------------------------------------------------------------------

# === Deterministic test inputs ===

# Simple deterministic state (21x1)
import numpy as np

# Use same random seed as MATLAB for reproducibility
np.random.seed(42)

# === Define data type ===
DTYPE = np.float64

# === Initial State Vector ===
state = np.array([
    1, 0, 0, 0,       # quaternion
    45.0, 0.0, 100.0, # lat (deg), lon (deg), alt (m)
    10.0, 5.0, -1.0,  # velocities (vn, ve, vd)
    0.001, -0.002, 0.0005,  # gyro bias
    0.0002, 0.0001, -0.0003, # accel bias
    0.01, -0.02, 0.005,      # gyro scale factor
    0.002, -0.001, 0.004     # accel scale factor
], dtype=DTYPE)

# === Earth constants ===
we = DTYPE(7.292115e-5)
a = DTYPE(6378137.0)
e2 = DTYPE(6.69437999014e-3)

# === Example measured vectors (modified) ===
a_meas = np.array([0.05, -0.08, 9.79], dtype=DTYPE)
w_meas = np.array([0.015, 0.018, -0.025], dtype=DTYPE)

# === Simple direction cosine matrix (identity, meaning b-frame = n-frame) ===
DCM_b2n = np.eye(3, dtype=DTYPE)

# === Covariance Matrices (modified) ===

# Initial covariance P — diagonal dominant, slightly random symmetric
base_P = 1e-3
P = np.diag(np.linspace(base_P, base_P * 5, 21))
P += 1e-5 * (np.random.rand(21, 21) - 0.5)
P = 0.5 * (P + P.T)  # enforce symmetry

# Process noise covariance Q — varied scales like MATLAB
gyro_white = 5e-6
accel_white = 2e-5
gyro_rw = 1e-7
accel_rw = 2e-6

Q = np.diag(np.array([
    gyro_white * 1.0, gyro_white * 0.8, gyro_white * 1.2,
    accel_white * 0.9, accel_white * 1.1, accel_white * 1.0,
    gyro_rw * 1.0, gyro_rw * 0.9, gyro_rw * 1.1,
    accel_rw * 1.0, accel_rw * 1.2, accel_rw * 0.8
], dtype=DTYPE))

# === Compute Pdot ===
Pdot = compute_Pdot(state, P, Q, a_meas, w_meas, DCM_b2n, we, a, e2)

# === Print full 21x21 table ===
np.set_printoptions(precision=6, suppress=True, linewidth=200)
print("=== Pdot (21x21) ===")
print(Pdot)

# === Save to CSV for MATLAB cross-validation ===
np.savetxt("Pdot_python_modified.csv", Pdot, delimiter=",", fmt="%.9e")
print("\n✅ Saved as 'Pdot_python_modified.csv' for cross-checking with MATLAB output.")
