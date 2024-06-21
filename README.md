# Introduction
This GitHub repository contains all the data and source code required to replicate the analysis and figures in our publication *Influence of Crossover on Capacity Fade of Symmetric Redox Flow Cells*.
These are the steps to replicate the calculations on your own computer.

### Clone the Repository from GitHub
    git clone https://github.com/AzizGordonEchem/2024-06-Xover-Fade.git

### Explanation of data organization
The top level directory /data includes the results of crossover experiments in Excel files (.xls).
The two CSV files capacity.csv and concentration.csv are processed data from cell cycling experiments done with UV-Vis spectrophotometry.
capacity.csv counts the charge moved in coulombs during 8 half cycles of charge / discharge.
concentration.csv has a calculation of the dimer concentration from the absorbance data.

The top level directory jupyter includes a Jupyter notebook that anlyzed the cell data.
It also includes two python scripts that can be run to obtain the cell cycle poitns and plot the data.
The subdirectory raw_data contains the raw cell cycling data and UV viz data as obtained directly from the scientific instruments.

The top level directory src contains Python programs that can be run from the command line to repeat the dimer rate constant calculation.

### Create an Anaconda environment with the necessary packages
Install Anaconda if it is not already on your system. 
See https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Then run these commands in a terminal if you are on a Windows platform:
    > cd \your\path\to\xover-fade
    > conda env create -f src/environment_windows.yml
    > conda activate xofade

This will create a new conda environment on your system named qefsm that includes all the packages required to run the Python programs.
It has not been tested on a Linux or Mac platform. However, in principle it should be easy to adapt to these platforms if required.

### Replicate the Dimer Rate Constant Calculation
To replicate the calculation of the dimer rate constant on your computer, run these steps in order:
    (xofade) python src/dimer_rate.py
    (xofade) python src/rfb_sim.py
