import numpy as np
import numpy.typing as npt

from data import load_capacity, load_conc_dimer
from analysis import calc_cycle_number, calc_soc, calc_aqds_conc, inst_eq, \
    spline_soc, integrate_conc, sweep_rate_conc
from plot_extra import plot_capacity, plot_conc_dimer, plot_conc_aqds, \
    plot_dimer_pred, plot_rate_rms, plot_rate_corr, plot_soc
from plot_main import plot_dimer_conc, plot_rate_sweep

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# The equilibrium constant for the dimer formation
K: float = 80.0

def main(do_sweep: bool, do_main_plot: bool, do_extra_plot: bool):
    """Estimate the rate of dimer formation from the data"""
    # The time associated with the capacity data; in seconds; shape (n_q,)
    t_q: NumpyFloatArray
    # The capacity data; amount of charge in coulombs moved in each half cycle; shape (n_q,)
    q: NumpyFloatArray
    # Load the capacity data
    t_q, q = load_capacity()
    # Get the maximum time where the capacity data is valid
    t_max: float = t_q[-1]

    # The time associated with the dimer concentration data; in seconds; shape (n_c,)
    t_c: NumpyFloatArray
    # The dimer concentration in molar, i.e. [QHQ] in Molar; shape (n_c,)
    c_QHQ: NumpyFloatArray
    # Load the dimer concentration data; shape (n_c,)
    t_c, c_QHQ = load_conc_dimer(t_max=t_max)

    # The cycle number for the capacity data; shape (n_q,)
    k_q: NumpyInt16Array
    # The cycle roll times; shape (n_cycle+1,)
    t_cycle: NumpyFloatArray
    # Calculate the cycle number and efficiency from the capacity data
    k_q, t_cycle = calc_cycle_number(t_q=t_q, q=q)

    # Calculate the state of charge aligned with the charge data
    s_q: NumpyFloatArray = calc_soc(t_q=t_q, q=q, t_c=t_c, c_QHQ=c_QHQ)

    # Calculate the concentration of all three AQDS species
    conc: NumpyFloatArray = calc_aqds_conc(t_q=t_q, s_q=s_q, c_QHQ=c_QHQ, t_c=t_c)
    # Unpack the concentrations
    c_Q: NumpyFloatArray = conc[:, 0]
    c_QH: NumpyFloatArray = conc[:, 1]

    # The time step for the integration; in seconds
    dt: float = 0.25
    # The time sample points for integration; use a fixed time step of dt
    t_i: NumpyFloatArray = np.arange(t_c[0], t_c[-1] + dt, dt)

    # The splined concentration at the time integration points
    # conc_i: NumpyFloatArray = spline_conc(t_c=t_c, conc=conc, t_i=t_i)

    # The splined SOC at the time integration points
    s_i: NumpyFloatArray = spline_soc(t_q=t_q, s_q=s_q, t_i=t_i)

    # The instantaneous equilibrium concentration of dimer
    c_QHQ_eq: NumpyFloatArray = inst_eq(c_Q=c_Q, c_QH=c_QH, K=K)

    # Do the rate sweep if requested
    if do_sweep:
        # Minimum and maximum values for the rate of dimer formation
        r_min: float = 1.0E-4
        r_max: float = 1.0E-2
        # Number of points to sweep for the rate of dimer formation
        n_r: int = 81
        # Rate constants to sweep
        r_sweep: NumpyFloatArray = np.geomspace(r_min, r_max, n_r)
        # RMS error between the predicted and actual dimer concentration
        rms: NumpyFloatArray
        # Correlation between the predicted and actual dimer concentration
        corr: NumpyFloatArray
        # Sweep using integration with known [Q] and [QH] to predict dimer
        # rms, corr = sweep_rate_dimer(t_i=t_i, conc_i=conc_i, t_c=t_c, c_QHQ=c_QHQ, r_sweep=r_sweep, K=K)
        # Sweep using integration with known SOC to predict dimer
        rms, corr = sweep_rate_conc(t_i=t_i, s_i=s_i, t_c=t_c, c_QHQ=c_QHQ, r_sweep=r_sweep, K=K)

        # Find the rate with the minimum RMS error
        i_opt = int(np.argmin(rms))
        r_opt = r_sweep[i_opt]
        err_opt = rms[i_opt]
        corr_opt = corr[i_opt]
        # Forward rate constant at this rate
        kf: float = K * r_opt
        # Reverse rate constant at this rate
        kr: float = r_opt

        # Report results
        print('Rate constant with maximum correlation:')
        print(f'kf   = {kf:8.3e} (M^-1 s^-1)')
        print(f'kr   = {kr:8.3e} (s^-1)')
        print(f'K_eq = {K:8.3f} (M^-1)')
        print(f'RMS error  : {err_opt:8.6f}')
        print(f'Correlation: {corr_opt:0.6f}')

        # Plot rate sweep results if requested
        if do_extra_plot:
            plot_rate_rms(r_sweep=r_sweep, rms=rms, i_opt=i_opt, K=K)
            plot_rate_corr(r_sweep=r_sweep, corr=corr, i_opt=i_opt, K=K)

    # The rate of dimer formating in the manuscript
    # r_ms: float = 0.03 / K
    # The rate from a previous run of the sweep
    r_prev: float = pow(10.0, -2.750)
    # The rate of dimer formation; in second^-1; set to the optimal value from the sweep when applicable
    r: float = r_opt if do_sweep else r_prev

    # Integrate the ODE
    conc_pred: NumpyFloatArray = integrate_conc(t_i=t_i, s_i=s_i, t_c=t_c, K=K, r=r)
    # Sample the dimer concentration at the concentration times
    pred: NumpyFloatArray = conc_pred[:, 2]

    # Build the main plots if requested
    if do_main_plot:
        # Plot measured and predicted dimer concentration
        plot_dimer_conc(t_c=t_c, c_QHQ=c_QHQ, pred=pred)
        # Plot sweep of rate constants
        plot_rate_sweep(r_sweep=r_sweep, rms=rms, corr=corr, i_opt=i_opt, K=K)
    
    # Build the extra plots if requested
    if do_extra_plot:
        # Plot the capacity data
        plot_capacity(t_q=t_q, q=q, k_q=k_q)

        # Plot the state of charge
        plot_soc(t_q=t_q, s_q=s_q)

        # Plot the dimer concentration data
        plot_conc_dimer(t_c=t_c, c_QHQ=c_QHQ)

        # Plot all the AQDS concentration data
        plot_conc_aqds(t_c=t_c, conc=conc, t_cycle=t_cycle)

        # Plot the experimental vs. predicted dimer concentration
        plot_dimer_pred(t_c=t_c, c_QHQ=c_QHQ, pred=pred, c_QHQ_eq=c_QHQ_eq)

if __name__ == '__main__':
    # Should we do the rate sweep?
    do_sweep: bool = True
    # Should we build the main plots for the manuscript?
    do_main_plot: bool = True
    # Should be build the extra plots?
    do_extra_plot: bool = False
    # Run the main function
    main(do_sweep=do_sweep, do_main_plot=do_main_plot, do_extra_plot=do_extra_plot)
