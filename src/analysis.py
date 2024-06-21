import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline, interp1d

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# Physical constants

# Faraday's constant in Coulombs per mole
F: float = 96485.33212  

# Experimental setup

# Volume in liters
vol: float = 5.0E-3
# Concentration of AQDS (total) in molar
aqds0: float = 0.10
# Number of electrons transferred in the reaction
n_e: int = 2
# Theoretical maximum capacity in Coulombs
q_max: float = n_e * F * aqds0 * vol

def calc_cycle_number(t_q: NumpyFloatArray, q: NumpyFloatArray) -> \
    tuple[NumpyInt16Array, NumpyFloatArray]:
    """
    Assign the cycle number based on the capacity data
    Inputs:
    t_q: time in seconds for charge data; shape (n_q,)
    q: capacity in Coulombs; shape (n_q,)
    Returns:
    k_q: cycle number for charge data; shape (n_q,)
    t_cycle: cycle change times; shape (n_cycle+1,)
    """
    # Number of data points
    n_q: int = len(q)
    # Array to store the cycle number of each data point aligned with the capacity data
    k_q: NumpyInt16Array = np.zeros(n_q, dtype=np.int16)

    # Locate the start of each cycle; rows where the capacity is zero
    start_idx = np.argwhere(q == 0.0).flatten()
    # Number of cycles
    n_cycle: int = len(start_idx)
    # Array to store times of cycle changes
    t_cycle: NumpyFloatArray = np.zeros(n_cycle+1, dtype=np.float64)

    # Assign cycle numbers aligned with the charge/discharge cycles
    for k in range(n_cycle):
        # Is this a "regular" cycle, i.e. not the last cycle?
        is_reg_cycle: bool = (k+1) < n_cycle
        # Start index for this cycle
        i0 = start_idx[k]
        # End index for this cycle; special handling for last cycle
        i1 = start_idx[k + 1] if is_reg_cycle else n_q
        # Assign cycle number aligned with capacity data
        k_q[i0:i1] = k
        # Start time for this cycle
        t0: float = float(t_q[i0])
        # End time for this cycle; special handling for last cycle
        t1: float = float(t_q[i1]) if is_reg_cycle else float(t_q[i1 - 1]) + 1.0
        # Save cycle roll times
        t_cycle[k] = t0
        if not is_reg_cycle:
            t_cycle[k+1] = t1

    return k_q, t_cycle

def calc_soc(t_q: NumpyFloatArray, q: NumpyFloatArray, 
             t_c: NumpyFloatArray, c_QHQ: NumpyFloatArray) -> NumpyFloatArray:
    """
    Calculate the SOC from the charge data, at time points aligned with t_q.
    Inputs:
    t_q: time in seconds for charge data; shape (n_q,)
    q: capacity in Coulombs; shape (n_q,)
    t_c: time in seconds for concentration data; shape (n_c,)
    c_QHQ: concentration of the dimer [QHQ] in Molar; shape (n_c,)
    """
    # Starting index for charge cycles
    ii0: NumpyInt16Array = np.argwhere(q == 0.0).flatten()
    # Ending index for charge cycles; NOT the index of the next start, but the last index in the cycle
    ii1: NumpyInt16Array = np.append(ii0[1:], len(q)) - 1
    # Number of charge cycles
    n_cycle: int = ii0.shape[0]
    # Array to store the SOC
    s_q: NumpyFloatArray = np.zeros_like(q)

    # Build a spline to interpolate the dimer concentration at a given time
    spline_QHQ: CubicSpline = CubicSpline(t_c, c_QHQ, axis=0)

    # Use the following variables for fractions of AQDS concentration
    # x = [Q]   / aqds0 - fraction total AQDS that is oxidized
    # y = [QH]  / aqds0 - fraction total AQDS that is reduced
    # z = [QHQ] / aqds0 - fraction total AQDS that is dimerized
    # conservation of AQDS: x + y + 2z = 1
    # definition of SOC: s = y + z
    # At all the points, z is known by splining the experimental [QHQ] data.
    # At the start of a charge cycle, there is no reduced AQDS. 
    # Therefore y=0, and x = 1 - 2z. So the SOC is z.
    # At the end of a charge cycle, there is no oxidized AQDS.
    # Therefore x=0, and y = 1 - 2z. So the SOC is 1 - z.

    # Calculate the SOC for each charge cycle
    for k in range(n_cycle):
        # Start and end indices for this cycle
        i0: int = ii0[k]
        i1: int = ii1[k]
        # Start and end times for this cycle
        t0: float = t_q[i0]
        t1: float = t_q[i1]
        # Start and end z for this cycle
        z0: float = float(spline_QHQ(t0)) / aqds0
        z1: float = float(spline_QHQ(t1)) / aqds0
        # Is this a charge cycle?
        is_charge: bool = (k % 2) == 0
        # Start and end SOC for this cycle
        s0: float = z0 if is_charge else (1.0 - z0)
        s1: float = (1.0 - z1) if is_charge else z1
        # How much did the soc change during the cycle?
        ds: float = s1 - s0
        # What was the charge at the end of the cycle?
        q_end: float = float(q[i1])
        # Slice for this cycle
        slc: slice = slice(i0, i1+1)
        # Interpolate the SOC for this cycle proportionally to coulombs moved
        s_q[slc] = s0 + q[slc] * (ds / q_end)
    return s_q

def calc_soc_direct(q: NumpyFloatArray) -> NumpyFloatArray:
    """
    Calculate the SOC directly from the charge data, by counting Coulombs.
    Inputs:
    t_q: time in seconds for charge data; shape (n_q,)
    q: capacity in Coulombs; shape (n_q,)
    """
    # Starting index for charge cycles
    ii0: NumpyInt16Array = np.argwhere(q == 0.0).flatten()
    # Ending index for charge cycles
    ii1: NumpyInt16Array = np.append(ii0[1:], len(q))
    # Number of charge cycles
    n_cycle: int = ii0.shape[0]
    # Array to store the SOC
    s_q: NumpyFloatArray = np.zeros_like(q)
    # SOC at the end of the last cycle
    s_last: float = 0.0
    # Calculate the SOC for each charge cycle
    for k in range(n_cycle):
        # Start and end indices for this cycle
        i0: int = ii0[k]
        i1: int = ii1[k]
        # Is this a charge cycle?
        is_charge: bool = (k % 2) == 0
        # Charge multiplier for this cycle
        charge_mult: float = 1.0 if is_charge else -1.0
        # Slice for this cycle
        slc: slice = slice(i0, i1)
        # SOC during this cycle
        s_q[slc] = s_last + charge_mult * q[slc] / q_max
        # Allowed range for SOC is [0, 1]
        s_q[slc] = np.clip(s_q[slc], 0.0, 1.0)
        # SOC at the end of this cycle
        s_last = s_q[i1-1]

    return s_q

def solve_eq(c_Q: NumpyFloatArray, c_QH: NumpyFloatArray, K: float) -> NumpyFloatArray:
    """    
    Solve for equilibrium concentration of the dimer [QHQ] given starting concentrations of 
    the oxidized and reduced forms of AQDS and the equilibrium constant.
    Inputs:
    c_Q: concentration of the oxidized form [Q] in Molar; shape (n_c,)
    c_QH: concentration of the reduced form [QH] in Molar; shape (n_c,)
    K: equilibrium constant in Molar^-1
    Returns:
    c_eq: equilibrium concentration of the dimer [QHQ] in Molar; shape (n_c,)
    """
    # quadratic equation coefficients
    p: NumpyFloatArray = -(c_Q + c_QH + 1.0 / K)
    q: NumpyFloatArray = c_Q * c_QH
    # solve the quadratic equation
    return 0.5 * (-p - np.sqrt(p**2 - 4.0 * q))

def inst_eq(c_Q: NumpyFloatArray, c_QH: NumpyFloatArray, K: float) -> NumpyFloatArray:
    """    
    The instantanous concentration of the dimer [QHQ] that would be in equilibrium
    with the oxidized and reduced concentrations given.
    Inputs:
    c_Q: concentration of the oxidized form [Q] in Molar; shape (n_c,)
    c_QH: concentration of the reduced form [QH] in Molar; shape (n_c,)
    K: equilibrium constant in Molar^-1
    Returns:
    c_eq: instantaneous equilibrium concentration of the dimer [QHQ] in Molar; shape (n_c,)
    """
    return K * c_Q * c_QH

def spline_conc(t_c: NumpyFloatArray, conc: NumpyFloatArray, t_i: NumpyFloatArray) -> NumpyFloatArray:
    """
    Spline the concentration data to a new set of time points
    Inputs:
    t_c: time in seconds for concentration data; shape (n_c,)
    conc: concentration data; shape (n_c, 3)
    t_i: time in seconds for interpolation; shape (n_t,)
    Returns:
    conc_i: splined concentration data; shape (n_t, 3)
    """
    # Build the spline
    conc_spline: CubicSpline = CubicSpline(t_c, conc, axis=0)
    # Evaluate the spline at the new time points
    return conc_spline(t_i).astype(np.float64)

def spline_soc(t_q: NumpyFloatArray, s_q: NumpyFloatArray, t_i: NumpyFloatArray) -> NumpyFloatArray:
    """
    Spline the state of charge data to a new set of time points
    Inputs:
    t_q: time in seconds for SOC data; shape (n_c,)
    s_q: state of charge data; shape (n_q,)
    t_i: time in seconds for interpolation; shape (n_t,)
    Returns:
    s_i: splined SOC data; shape (n_t, 3)
    """
    # Build the spline
    soc_spline: CubicSpline = CubicSpline(t_q, s_q, axis=0)
    # Evaluate the spline at the new time points
    return soc_spline(t_i).astype(np.float64)

def calc_aqds_conc(t_q: NumpyFloatArray, s_q: NumpyFloatArray, c_QHQ: NumpyFloatArray, t_c: NumpyFloatArray) \
    -> NumpyFloatArray:
    """
    Calculate the concentration of oxidized and reduced AQDS given dimer concentration and estimated SOC.
    Inputs:
    t_q: time in seconds for charge data; shape (n_q,)
    s_q: state of charge; shape (n_q,)
    c_QHQ: concentration of the dimer [QHQ] in Molar; shape (n_c,)
    t_c: time in seconds, aligned with the concentration data; shape (n_c,)
    Returns:
    conc: concentrations of [Q, QH, QHQ]; shape (n_c, 3)
    """
    # Number of data points
    n_c: int = t_c.shape[0]
    # Create an array to store the concentrations
    conc: NumpyFloatArray = np.zeros((n_c, 3))
    # Spline for the SOC
    soc_spline: CubicSpline = CubicSpline(t_q, s_q, axis=0)
    # The state of charge at the concentration time points
    s: NumpyFloatArray = soc_spline(t_c).astype(np.float64)
    # The fraction of AQDS that is dimerized
    z: NumpyFloatArray = c_QHQ / aqds0
    # The fraction of AQDS that is oxidized
    x: NumpyFloatArray = np.clip(1.0 - s - z, 0.0, 1.0)
    # The fraction of AQDS that is reduced
    y: NumpyFloatArray = np.clip(s - z, 0.0, 1.0)
    # Copy concentrations to the output array
    conc[:, 0] = x * aqds0
    conc[:, 1] = y * aqds0
    conc[:, 2] = z * aqds0
    return conc

def integrate_dimer(t_i: NumpyFloatArray, conc_i: NumpyFloatArray,
                   t_c: NumpyFloatArray, K: float, r: float) -> NumpyFloatArray:
    """
    Integrate the dimer concentration using fixed concentrations for oxidized and reduced species
    Inputs:
    t_i: time in seconds for integration time points; shape (n_t,)
    conc_i: splined concentration of all three species [Q], [QH], [QHQ] in Molar; shape (n_t, 3)
    t_c: time in seconds for concentration data; shape (n_c,)
    K: equilibrium constant in Molar^-1
    r: rate constant in s^-1
    Returns:
    pred: predicted dimer concentration in Molar; shape (n_c,)
    """
    # Unpack conc_i
    # The oxidized concentration splined at the integration time points; in Molar
    c_Q_i: NumpyFloatArray = conc_i[:, 0]
    # The reduced concentration splined at the integration time points; in Molar
    c_QH_i: NumpyFloatArray = conc_i[:, 1]

    # Forward rate constant
    kf: float = K * r
    # Reverse rate constant
    kr: float = r
    # Time step for integration; must be constant
    dt: float = float(t_i[1] - t_i[0])
    
    # Total number of time points for the integration
    n_t: int = len(t_i)
    # Array of integrated dimer concentration
    c_QHQ_i: NumpyFloatArray = np.zeros_like(t_i)
    # Initial condition - zero dimer concentration
    c_QHQ_i[0] = 0.0

    # Integrate the dimer concentration
    for i in range(n_t - 1):
        c_QHQ_i[i+1] = c_QHQ_i[i] + (kf * c_Q_i[i] * c_QH_i[i] - kr * c_QHQ_i[i]) * dt

    # Evaluate the integrated dimer concentration at the original time points
    pred: NumpyFloatArray = np.interp(t_c, t_i, c_QHQ_i).astype(np.float64)
    return pred

def integrate_conc(t_i: NumpyFloatArray, s_i: NumpyFloatArray, t_c: NumpyFloatArray,
                    K: float, r: float) -> NumpyFloatArray:
    """
    Integrate the concentrations of all three AQDS species using the SOC and rate constants.
    Inputs:
    t_i: time in seconds for integration time points; shape (n_t,)
    s_i: splined state of charge at integration time points; shape (n_t,)
    t_c: time in seconds for concentration data; shape (n_c,)
    K: equilibrium constant in Molar^-1
    r: rate constant in s^-1
    Returns:
    conc: predicted species concentrations in Molar; shape (n_c, 3,)
    """
    # Forward rate constant
    kf: float = K * r
    # Reverse rate constant
    kr: float = r
    # Time step for integration; must be constant
    dt: float = float(t_i[1] - t_i[0])

    # Total number of time points for the integration
    n_t: int = len(t_i)
    # Array of integrated concentrations
    conc_i: NumpyFloatArray = np.zeros((n_t, 3))
    # Initial condition - all AQDS is oxidized
    conc_i[0,0] = aqds0
    # Integrate the ODE
    for i in range(n_t - 1):
        # Change of the dimer concentration in this time step
        dQHQ = (kf * conc_i[i,0] * conc_i[i,1] - kr * conc_i[i,2]) * dt
        # Amount of oxidized converted to reduced by SOC change in this time step
        dRed = (s_i[i+1] - s_i[i]) * aqds0
        # Update the concentrations
        conc_i[i+1, :] = conc_i[i, :] + np.array([-dRed - dQHQ, dRed - dQHQ, dQHQ])

    # Resample the concentrations at the original time points
    conc_spline: interp1d = interp1d(t_i, conc_i, axis=0, kind='linear')
    conc_c: NumpyFloatArray = conc_spline(t_c)
    return conc_c

def conc_rms(c_exp: NumpyFloatArray, c_pred: NumpyFloatArray) -> float:
    """
    Calculate the root mean square error between the experimental and predicted concentrations
    Inputs:
    c_exp: experimental concentration in Molar; shape (n_c,)
    c_pred: predicted concentration in Molar; shape (n_c,)
    Returns:
    rms: root mean square error
    """
    # Mean experimental dimer concentration
    c_mean: float = float(np.mean(c_exp))
    # Calculate the error; divide by c_mean to normalize
    err = (c_exp - c_pred) / c_mean
    # Return the RMS error
    return np.sqrt(np.mean(np.square(err)))

def conc_corr(c_exp: NumpyFloatArray, c_pred: NumpyFloatArray) -> float:
    """Calculate the correlation coefficient between the experimental and predicted concentrations
    Inputs:
    c_exp: experimental concentration in Molar; shape (n_c,)
    c_pred: predicted concentration in Molar; shape (n_c,)
    Returns:
    corr: correlation coefficient
    """
    # The correlation matrix
    corr_mat: NumpyFloatArray = np.corrcoef(c_exp, c_pred)
    # Return the correlation coefficient
    return corr_mat[0, 1]

def sweep_rate_dimer(t_i: NumpyFloatArray, conc_i: NumpyFloatArray, 
        t_c: NumpyFloatArray, c_QHQ: NumpyFloatArray, r_sweep: NumpyFloatArray, K: float, ) \
            -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """
    Sweep the rate constant and calculate the RMS error for each rate.
    Integration is done using the concentrations of the oxidized and reduced species.
    Return the rate with the minimum RMS error.
    Inputs:
    t_i: time in seconds for integration time points; shape (n_t,)
    conc_i: integrated concentrations ([Q], [QH], [QHQ]) in Molar; shape (n_t, 3,)
    t_c: time in seconds for concentration data; shape (n_c,)
    c_QHQ: concentration of the dimer [QHQ] in Molar; shape (n_c,)
    r_sweep: rate constants to sweep; shape (n_r,)
    K: equilibrium constant in Molar^-1
    Returns:
    rms: RMS error for each rate; shape (n_r,)
    corr: correleation between the experimental and predicted concentrations for each rate; shape (n_r,)
    """
    # Array to store the RMS errors
    rms: NumpyFloatArray = np.zeros_like(r_sweep)
    # Array to store the correlation coefficients
    corr: NumpyFloatArray = np.zeros_like(r_sweep)
    # Sweep the rate constants
    for i, r in enumerate(r_sweep):
        # Integrate the dimer concentration
        c_pred: NumpyFloatArray = integrate_dimer(t_i=t_i, conc_i=conc_i, t_c=t_c, K=K, r=r)
        # Calculate the RMS error
        rms[i] = conc_rms(c_exp=c_QHQ, c_pred=c_pred)
        # Calculate the correlation coefficient
        corr[i] = conc_corr(c_exp=c_QHQ, c_pred=c_pred)

    return rms, corr

def sweep_rate_conc(t_i: NumpyFloatArray, s_i: NumpyFloatArray, 
        t_c: NumpyFloatArray, c_QHQ: NumpyFloatArray, r_sweep: NumpyFloatArray, K: float, ) \
            -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """
    Sweep the rate constant and calculate the RMS error for each rate.
    Integration is done using the SOC and rate constants.
    Return the rate with the minimum RMS error.
    Inputs:
    t_i: time in seconds for integration time points; shape (n_t,)
    s_i: interpolated state of charge; shape (n_t,)
    t_c: time in seconds for concentration data; shape (n_c,)
    c_QHQ: concentration of the dimer [QHQ] in Molar; shape (n_c,)
    r_sweep: rate constants to sweep; shape (n_r,)
    K: equilibrium constant in Molar^-1
    Returns:
    rms: RMS error for each rate; shape (n_r,)
    corr: correleation between the experimental and predicted concentrations for each rate; shape (n_r,)
    """
    # Array to store the RMS errors
    rms: NumpyFloatArray = np.zeros_like(r_sweep)
    # Array to store the correlation coefficients
    corr: NumpyFloatArray = np.zeros_like(r_sweep)
    # Sweep the rate constants
    for i, r in enumerate(r_sweep):
        # Integrate the concentrations
        conc_pred: NumpyFloatArray = integrate_conc(t_i=t_i, s_i=s_i, t_c=t_c, K=K, r=r)
        # Extract the dimer concentration
        c_pred = conc_pred[:, 2]
        # Calculate the RMS error
        rms[i] = conc_rms(c_exp=c_QHQ, c_pred=c_pred)
        # Calculate the correlation coefficient
        corr[i] = conc_corr(c_exp=c_QHQ, c_pred=c_pred)

    return rms, corr
