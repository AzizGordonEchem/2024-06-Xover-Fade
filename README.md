# Introduction
This GitHub repository contains all the data and source code required to replicate the analysis and figures in our publication *Influence of Crossover on Capacity Fade of Symmetric Redox Flow Cells*.
These are the steps to replicate the calculations on your own computer.

### Clone the Repository from GitHub
    git clone https://github.com/AzizGordonEchem/2024-06-Xover-Fade.git

### Create an Anaconda environment with the necessary packages
Install Anaconda if it is not already on your system. 
See https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Then run these commands in a terminal if you are on a Windows platform:
    > cd \your\path\to\xover-fade
    > conda env create -f src/environment_windows.yml
    > conda activate xofade

This will create a new conda environment on your system named qefsm that includes all the packages required to run the Python programs.
It has not been tested on a Linux or Mac platform. However, in principle it should be easy to adapt to these platforms if required.

### Run the Python scripts in order
    (xofade) python src/dimer_rate.py
    (xofade) python src/rfb_sim.py
