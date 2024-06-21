import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path

# Path for data directory
data_dir = Path('data')

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]
NumpyBoolArray = npt.NDArray[np.bool_]

def load_capacity() -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """
    Load the capacity data. Returns t_q, cap.
    t_q: time in seconds
    cap: capacity in Coulombs
    """
    # Load the data as a pandas DataFrame
    df: pd.DataFrame = pd.read_csv(data_dir / 'capacity.csv')
    # Extract the time as a numpy array; time is in minutes
    t_q: NumpyFloatArray = df['Time'].to_numpy()
    # Offset the time data so the first data point is at t=0; this is per Eric Fell's email 2024-06-11-1337
    t_q -= t_q[0]
    # Convert the time to seconds so rate constant has correct units
    t_q *= 60.0
    # Extract the capacity as a numpy array; "capacity" is the total charge in coulombs
    cap: NumpyFloatArray = df['Capacity'].to_numpy()
    return t_q, cap

def load_conc_dimer(t_max: float) -> tuple[NumpyFloatArray, NumpyFloatArray]:
    """
    Load the dimer concentration data. Returns t_c, c_QHQ.
    t_c: time in seconds
    c_QHQ: concentration of the dimer [QHQ] in Molar
    """
    # Load the data as a pandas DataFrame
    df: pd.DataFrame = pd.read_csv(data_dir / 'concentration.csv')
    # Extract the time as a numpy array
    t_c: NumpyFloatArray = df['Time'].to_numpy()
    # Convert the time to seconds so rate constant has correct units
    t_c *= 60.0
    # Extract the concentration of the dimer [QHQ] in molar as a numpy array
    c_QHQ: NumpyFloatArray = df['DimerConc'].to_numpy()
    # The rows numbered from 987 to 1009 appear to be duplicates; remove them
    # t_c = t_c[:987]
    # c_QHQ = c_QHQ[:987]
    # Only take rows where the time is less than t_max
    mask: NumpyBoolArray = (t_c <= t_max)
    t_c = t_c[mask]
    c_QHQ = c_QHQ[mask]
    return t_c, c_QHQ

