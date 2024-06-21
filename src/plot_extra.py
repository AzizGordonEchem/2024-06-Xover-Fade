import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path

# Path for figures directory
plot_dir = Path('figs_extra')

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# Set Matplotlib parameters
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True

def plot_capacity(t_q: NumpyFloatArray, q: NumpyFloatArray, k_q: NumpyInt16Array) -> None:
    """
    Plot the capacity data.
    Inputs:
    t_q: time in seconds; shape (n_q,)
    q: capacity in Coulombs; shape (n_q,)
    k_q: cycle number; shape (n_q,)
    """
    # Convert the time from seconds to minutes for plotting
    t_min: NumpyFloatArray = t_q / 60.0
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Set the title and axis labels
    ax.set_title('Capacity vs. Time')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Capacity (Coulombs)')
    # Plot the capacity data with different colors for each cycle
    for k in np.unique(k_q):
        mask: NumpyBoolArray = (k_q == k)
        t_end: float = t_min[mask][-1]
        is_charge: bool = (k % 2 == 0)
        color: str = 'red' if is_charge else 'blue'
        label: str = f'Cycle {k:d}'
        ax.plot(t_min[mask], q[mask], color=color, label=label)
        ax.axvline(x=t_end, color='black', linestyle='--')
    # Save the plot
    fig.savefig(plot_dir / '01_capacity.png')

def plot_soc(t_q: NumpyFloatArray, s_q: NumpyFloatArray):
    """
    Plot concentration of all three AQDS species
    Inputs:
    t_c: time in seconds; shape (n_c,)
    soc: state of charge (net charge / theoretical max charge); shape (n_c,)
    """
    # Convert the time from seconds to minutes for plotting
    t_min: NumpyFloatArray = t_q / 60.0
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the concentrations
    ax.plot(t_min, s_q , color='blue', label='SOC (charge)')
    # Set the title and axis labels
    ax.set_title('State of Charge vs. Time')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('State of Charge (dimensionless)')
    # Add a legend
    ax.legend()
    # Save the plot
    fig.savefig(plot_dir / '02_soc.png')

def plot_conc_dimer(t_c: NumpyFloatArray, c_QHQ: NumpyFloatArray):
    """
    Plot the dimer concentration data
    Inputs:
    t_c: time in seconds; shape (n_c,)
    c_QHQ: concentration of the dimer [QHQ] in Molar; shape (n_c,)
    """
    # Convert the time from seconds to minutes for plotting
    t_min: NumpyFloatArray = t_c / 60.0
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the dimer concentration
    ax.plot(t_min, c_QHQ, color='blue', label='[QHQ]')
    # Set the title and axis labels
    ax.set_title('Dimer Concentration vs. Time')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Dimer Concentration [QHQ] (Molar)')
    # Add a legend
    ax.legend()
    # Save the plot
    fig.savefig(plot_dir / '03_dimer_conc.png')

def plot_conc_aqds(t_c: NumpyFloatArray, conc: NumpyFloatArray, t_cycle: NumpyFloatArray):
    """
    Plot concentration of all three AQDS species
    Inputs:
    t_c: time in seconds; shape (n_c,)
    conc: concentration of all three AQDS species; shape (n_c, 3)
    t_cycle: cycle roll times; shape (n_cycle,)
    """
    # Convert the time from seconds to minutes for plotting
    t_min: NumpyFloatArray = t_c / 60.0
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the concentrations
    ax.plot(t_min, conc[:,0] , color='blue', label='[Q]')
    ax.plot(t_min, conc[:,1] , color='red',  label='[QH]')
    ax.plot(t_min, conc[:,2] , color='purple', label='[QHQ]')
    # Plot the cycle roll times
    for t in t_cycle:
        ax.axvline(x=t / 60.0, color='black', linestyle='--')
    # Plot zero concentration
    ax.axhline(y=0.0, color='black', linestyle='--')
    # Set the title and axis labels
    ax.set_title('Concentration vs. Time')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (Molar)')
    # Add a legend
    ax.legend()
    # Save the plot
    fig.savefig(plot_dir / '04_aqds_conc.png')

def plot_dimer_pred(t_c: NumpyFloatArray, c_QHQ: NumpyFloatArray, pred: NumpyFloatArray, c_QHQ_eq: NumpyFloatArray):
    """
    Plot the dimer concentration data vs. the predictions
    Inputs:
    t_c: time in seconds; shape (n_c,)
    c_QHQ: concentration of the dimer [QHQ] in Molar; shape (n_c,)
    pred: predicted dimer concentration; shape (n_c,)
    c_QHQ_eq: dimer concentration at instantaneous equilibrium; shape (n_c,)
    """
    # Convert the time from seconds to minutes for plotting
    t_min: NumpyFloatArray = t_c / 60.0
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the experimental dimer concentration
    ax.plot(t_min, c_QHQ, color='blue', label='Data')
    # Plot the predicted dimer concentration
    ax.plot(t_min, pred, color='red', label='Calc')
    # The dimer concentration at instantaneous equilibrium given current [Q] and [QH]
    ax.plot(t_min, c_QHQ_eq, color='green', label='Eq')
    # Set the title and axis labels
    ax.set_title('Dimer Concentration - Actual vs. Calculated')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Dimer Concentration [QHQ] (Molar)')
    # Add a legend
    ax.legend()
    # Save the plot
    fig.savefig(plot_dir / '05_dimer_vs_calc.png')

def plot_rate_rms(r_sweep: NumpyFloatArray, rms: NumpyFloatArray, i_opt: int, K: float):
    """
    Plot the rate sweep data
    Inputs:
    r_sweep: Rate sweep values; shape (n_r,)
    rms: RMS error for each rate; shape (n_r,)
    i_opt: Index of rate chosen to be optimal
    """
    # Unpack the rate with the minimum RMS error
    r: float = r_sweep[i_opt]
    # Unpack the minimum RMS error
    rms_opt: float = rms[i_opt]
    # Calculate forward and reverse rate constants
    kf: float = K * r
    kr: float = r

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the RMS error vs. rate sweep
    ax.plot(r_sweep, rms, color='blue', marker='.', label='RMS Error')
    # Plot the minimum RMS error
    ax.axvline(x=r, color='red', linestyle='--', label=f'Min Error: {rms_opt:0.3f}')
    # X axis is on a log scale
    ax.set_xscale('log')
    # Set the title and axis labels
    ax.set_title('Rate Sweep - RMS Error')
    ax.set_xlabel(r'Rate of Dimer Formation ($s^{-1}$)')
    ax.set_ylabel('RMS Error (dimensionless)')
    # Add a legend
    ax.legend()
    # Label the minimum RMS error
    xy = (r, rms_opt)
    rms_min = np.min(rms)
    rms_max = np.max(rms)
    text_x = r * 1.1
    text_y = 0.93 * rms_min + 0.07 * rms_max
    xytext = (text_x, text_y)
    # arrowprops = dict(facecolor='black', arrowstyle='->')
    label = f'$k_f$ = {kf:6.3e}\n$k_r$ = {kr:6.3e}'
    ax.annotate(label, xy=xy, xytext=xytext)
    # Save the plot
    fig.savefig(plot_dir / '06_rms_err.png')

def plot_rate_corr(r_sweep: NumpyFloatArray, corr: NumpyFloatArray, i_opt: int, K: float):
    """
    Plot the rate sweep data
    Inputs:
    r_sweep: rate sweep values; shape (n_r,)
    corr: Correlation error for each rate; shape (n_r,)
    i_opt: Index of rate chosen to be optimal
    """
    # Unpack the rate with the minimum RMS error
    r: float = r_sweep[i_opt]
    # Unpack the maximum correlation
    corr_opt: float = corr[i_opt]
    # Calculate forward and reverse rate constants
    kf: float = K * r
    kr: float = r

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the RMS error vs. rate sweep
    ax.plot(r_sweep, corr, color='blue', marker='.', label='Correlation')
    # Plot the minimum RMS error
    ax.axvline(x=r, color='red', linestyle='--', label=f'Max Corr: {corr_opt:0.3f}')
    # X axis is on a log scale
    ax.set_xscale('log')
    # Set the title and axis labels
    ax.set_title('Rate Sweep - Correlation')
    ax.set_xlabel(r'Rate of Dimer Formation ($s^{-1}$)')
    ax.set_ylabel('Correlation to Data (dimensionless)')
    # Add a legend
    ax.legend()
    # Label the optimal correlation
    xy = (r, corr_opt)
    corr_min = np.min(corr)
    corr_max = np.max(corr)
    text_x = r * 1.1
    text_y = 0.97 * corr_min + 0.03 * corr_max
    xytext = (text_x, text_y)
    # arrowprops = dict(facecolor='black', arrowstyle='->')
    label = f'$k_f$ = {kf:6.3e}\n$k_r$ = {kr:6.3e}'
    ax.annotate(label, xy=xy, xytext=xytext)
    # Save the plot
    fig.savefig(plot_dir / '07_corr.png')

def plot_charge(t_q: NumpyFloatArray, q: NumpyFloatArray, k_q: NumpyInt16Array, q_max: float):
    """
    Plot the net charge data
    Inputs:
    t_q: time in seconds; shape (n_q,)
    q: net charge in Coulombs; shape (n_q,)
    k_q: cycle number; shape (n_q,)
    q_max: theoretical maximum capacity in Coulombs; scalar
    """
    # Convert the time from seconds to minutes for plotting
    t_min: NumpyFloatArray = t_q / 60.0
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Set the title and axis labels
    ax.set_title('Net Charge vs. Time')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Net Charge (Coulombs)')
    # Plot the capacity data with different colors for each cycle
    for k in np.unique(k_q):
        mask: NumpyBoolArray = (k_q == k)
        is_charge: bool = (k % 2 == 0)
        color: str = 'red' if is_charge else 'blue'
        label: str = f'Cycle {k:d}'
        ax.plot(t_min[mask], q[mask], color=color, label=label)
        # ax.axvline(x=t_end, color='black', linestyle='--')
    # Theoretical maximum capacity
    ax.axhline(y=q_max, color='black', linestyle='--', label='Theoretical')
    # Save the plot
    fig.savefig(plot_dir / '08_charge.png')
