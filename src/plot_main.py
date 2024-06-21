import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path

# Path for figures directory
plot_dir = Path('figs')

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

# Set Matplotlib parameters
# plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Helvetica'

def set_plot_params() -> None:
    """Set plot parameters on the active plot to match style of the manuscript"""
    # fontsize of the x and y labels (words)
    plt.rc('axes', linewidth=1.2, labelsize=10)   
    # fontsize of the tick labels (numbers) 
    plt.rc('xtick', labelsize=9)                  
    # fontsize of the tick labels
    plt.rc('ytick', labelsize=9)                  
    # legend fontsize
    plt.rc('legend', fontsize=8)
    # Set the tick parameters
    x_ticks = { 
    "top" : True, 
    "direction" : "in", 
    "minor.visible" : True,  
    "major.size" : 4, 
    "major.width" :  1.2, 
    "minor.size" : 2, 
    "minor.width" : 0.5}
    y_ticks = x_ticks.copy()
    y_ticks["right"] = y_ticks.pop("top")
    # Bind the tick parameters to the plot settings
    plt.rc('xtick', **x_ticks)
    plt.rc('ytick', **y_ticks)
    # Legend fontsize
    plt.rc('legend', fontsize=8)

def plot_dimer_conc(t_c: NumpyFloatArray, c_QHQ: NumpyFloatArray, pred: NumpyFloatArray) -> None:
    """
    Plot the dimer concentration data vs. the predictions
    Inputs:
    t_c: time in seconds; shape (n_c,)
    c_QHQ: concentration of the dimer [QHQ] in Molar; shape (n_c,)
    pred: predicted dimer concentration; shape (n_c,)
    """
    # Convert the time from seconds to minutes for plotting
    t_min: NumpyFloatArray = t_c / 60.0
    # Set the plot parameters
    set_plot_params()
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    # Plot the experimental dimer concentration
    ax.plot(t_min, c_QHQ, color='purple', label='Data', linewidth=2.00)
    # Plot the predicted dimer concentration
    ax.plot(t_min, pred, color='green', label='Calc', linewidth = 1.00)
    # Set the title and axis labels
    ax.set_title('Dimer Concentration - Actual vs. Calculated')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Dimer concentration (M)')
    # Add a legend
    ax.legend()
    # Save the plot
    fig.savefig(plot_dir / '01_dimer_vs_calc.png')

def plot_rate_sweep(r_sweep: NumpyFloatArray, rms: NumpyFloatArray, corr: NumpyFloatArray, i_opt: int, K: float):
    """
    Plot the rate sweep data
    Inputs:
    r_sweep: Rate sweep values; shape (n_r,)
    rms: RMS error for each rate; shape (n_r,)
    i_opt: Index of rate chosen to be optimal
    """
    # Convert rate into units of kf to match the manuscript
    kf = K * r_sweep
    # Unpack the rate with the minimum RMS error
    kf_opt: float = kf[i_opt]
    # Unpack the minimum RMS error and correlation
    rms_opt: float = rms[i_opt]
    corr_opt: float = corr[i_opt]

    # Create a figure and axis
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8.0, 4.0))
    fig.suptitle('Best Fit Dimer Forward Rate $k_f$')
    # Set the plot parameters
    set_plot_params()

    # Plot the RMS error vs. rate sweep
    ax1.plot(kf, rms, color='blue', marker='.', label='RMS Error')
    # Plot the minimum RMS error
    ax1.axvline(x=kf_opt, color='red', linestyle='--', label=f'Min {rms_opt:0.3f}')
    # X axis is on a log scale
    ax1.set_xscale('log')
    # Set the title and axis labels
    ax1.set_xlabel(r'Forward Rate Constant $k_f (M^{-1} \cdot s^{-1}$)')
    ax1.set_ylabel('RMS Error vs. Data (dimensionless)')
    # Add a legend
    ax1.legend(loc='lower left')
    # Label the minimum RMS error
    xy = (kf_opt, rms_opt)
    rms_min = np.min(rms)
    rms_max = np.max(rms)
    text_x = kf_opt * 1.1
    text_y = 0.88 * rms_min + 0.12 * rms_max
    xytext = (text_x, text_y)
    label = f'$k_f$ = {kf_opt:0.3f}'
    ax1.annotate(label, xy=xy, xytext=xytext)

    # Plot the correlation vs. rate sweep
    ax2.plot(kf, corr, color='blue', marker='.', label='Correlation')
    # Plot the minimum RMS error
    ax2.axvline(x=kf_opt, color='red', linestyle='--', label=f'Max {corr_opt:0.3f}')
    # X axis is on a log scale
    ax2.set_xscale('log')
    # Set the title and axis labels
    ax2.set_xlabel(r'Forward Rate Constant $k_f (M^{-1} \cdot s^{-1}$)')
    ax2.set_ylabel('Correlation to Data (dimensionless)')
    # Add a legend
    ax2.legend(bbox_to_anchor=(0.12, 0.02), loc='lower left')
    # Label the minimum RMS error
    xy = (kf_opt, corr_opt)
    corr_min = np.min(corr)
    corr_max = np.max(corr)
    text_x = kf_opt * 1.1
    text_y = 0.97 * corr_min + 0.03 * corr_max
    xytext = (text_x, text_y)
    label = f'$k_f$ = {kf_opt:0.3f}'
    ax2.annotate(label, xy=xy, xytext=xytext)

    # Save the plot
    fig.savefig(plot_dir / '02_rate_sweep.png')