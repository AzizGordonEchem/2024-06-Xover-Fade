from rfbzero.redox_flow_cell import ZeroDModel
from rfbzero.experiment import ConstantCurrentConstantVoltage
from rfbzero.crossover import Crossover
from rfbzero.degradation import (ChemicalDegradationReduced,
                                 Dimerization,
                                 MultiDegradationMechanism)
                                                                            

# experimentally determined values for NR211, NR212, N115, N117
thickness = [25, 50, 125, 183]                      # microns
permeability = [8.3e-9,  6.7e-9,  1.6e-9,  2.3e-9]  # cm^2/s
ohmic_resistance = [0.035, 0.038, 0.060, 0.075]     # ohms


# initialize first-order chemical degradation
chem_deg = ChemicalDegradationReduced(rate_order=1, rate_constant=1.0e-8)

# set dimer formation rate constants
K_dimer = 75                      # 1/M
k_forward = 0.03                  # (1/M)/s
k_backward = k_forward / K_dimer  # 1/s


for t,p,r in zip(thickness, permeability, ohmic_resistance):
    # define dimerization in CLS and NCLS
    dimer_cls = Dimerization(forward_rate_constant=k_forward,
                             backward_rate_constant=k_backward,
                             )
    dimer_ncls = Dimerization(forward_rate_constant=k_forward,
                              backward_rate_constant=k_backward,
                              )

    # include chemical degradation and dimerization mechanisms in CLS and NCLS
    multi_cls = MultiDegradationMechanism([chem_deg, dimer_cls])
    multi_ncls = MultiDegradationMechanism([chem_deg, dimer_ncls])

    # define the symmetric cell setup
    cell = ZeroDModel(volume_cls=0.005,      # L
                      volume_ncls=0.010,     # L
                      c_ox_cls=0.05,         # M
                      c_red_cls=0.05,        # M
                      c_ox_ncls=0.05,        # M
                      c_red_ncls=0.05,       # M
                      ocv_50_soc=0.0,        # V
                      resistance=r,          # ohms
                      k_0_cls=1e-3,          # cm/s
                      k_0_ncls=1e-3,         # cm/s
                      time_step=0.05,        # sec
                      num_electrons_cls=2,   # electrons
                      num_electrons_ncls=2,  # electrons
                      )

    # define the CCCV protocol
    protocol = ConstantCurrentConstantVoltage(voltage_limit_charge=0.2,        # V
                                              voltage_limit_discharge=-0.2,    # V
                                              current_cutoff_charge=0.005,     # A
                                              current_cutoff_discharge=-0.005, # A
                                              current=0.05,                    # A
                                              )

    # define the crossover mechanism
    cross = Crossover(membrane_thickness=t, permeability_ox=p, permeability_red=p)
    
    # putting it all together
    all_results = protocol.run(cell_model=cell,
                               duration=90000,  # cycle time to simulate (s)
                               cls_degradation=multi_cls,
                               ncls_degradation=multi_ncls,
                               crossover=cross,
                               )
