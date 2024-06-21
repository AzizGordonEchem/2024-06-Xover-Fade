"""
Created on Fri Feb  4 11:46:07 2022

@author: Eric Fell
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


class cycle_data_selector:
    """
    Takes input folder and Novonix/BioLogic cycling .csv/.mpt file and 
    extracts cycle info as desired 
    
    can return:
        'time', 'step_time', 'discharge_capacity', 'charge_capacity', 
        'capacity', 'voltage', 'current', 'dQdV'
    """

    def __init__(self, cycles, folder, filename, plot=True, dV_threshold=0.00055):
        """Input file location of Novonix .csv file."""
        self.cycle_list = cycles
        self.location = folder
        self.file = filename
        self.plot = plot
        self.dV_threshold = dV_threshold
        
        assert filename.endswith(".csv") or filename.endswith(".mpt"), (
            "Filetype not supported")
        if filename.endswith('.csv'):
            assert not filename.endswith('_out.csv'), (
                "Not a raw data Novonix file")
        elif filename.endswith('.mpt'):
            assert not filename.endswith('_out.mpt'), (
                "Not a raw data BioLogic file")
        assert dV_threshold > 0.0, "dV_threshold cannot be negative Volts"
        assert dV_threshold < 0.1, "dV_threshold must be less than 0.1 Volts"
        if filename.endswith('.csv'):
            self.is_novonix = True
        else:
            self.is_novonix = False

    ######################################################
    def select_cycles(self):

        if self.file.endswith(".csv"): # Novonix
            col_position = [1, 2, 3, 4, 5, 6, 7, -2]
            ch_cycles = [(2*i) + (i - 1) for i in self.cycle_list] # how Novonix handles cycle numbering
            disch_cycles = [3*i for i in self.cycle_list]
            delim = ','
            keyword = 'Date and Time'
            idx_cycle = 0
            idx_step = -1
            cols = ['cycle_number', 'step', 'time', 'step_time', 'current', 
                    'voltage', 'capacity', 'half_cycle']
        else: # BioLogic
            ch_cycles = [(2*i) for i in self.cycle_list] # how BioLogic handles half-cycle numbering
            disch_cycles = [(2*i) + 1 for i in self.cycle_list]
            delim = '\t'
            keyword = 'mode'
            idx_cycle = -1
            idx_step = 3
            need_cols = ['time/s', 'control/V', '<I>/mA', 'half cycle', 
                         'Q discharge/mA.h', 'Q charge/mA.h', 'Capacity/mA.h', 
                         'cycle number']
            cols = ['time', 'voltage', 'current', 'half_cycle', 
                    'discharge_capacity', 'charge_capacity', 'capacity', 
                    'cycle_number']
        #
        cycles = ch_cycles + disch_cycles
        cycles.sort()
        
        read_lines = False
        cycle_data = []
        with open(self.location + '\\' + self.file) as f:
            csv_reader = csv.reader(f, delimiter=delim)
            for count, row in enumerate(csv_reader):
                # first determine number of lines to skip before data appears in file
                if (len(row) > 0) and (row[0] == keyword):
                    if self.file.endswith('.csv'):
                        read_lines = True
                        continue
                    else: # biologic, done this way because some files have dif ordered headers
                        col_position = [row.index(i) for i in need_cols]
                        read_lines = True
                        continue
                # now going through data
                if read_lines:
                    cycle_number = int(float(row[col_position[idx_cycle]]))
                    step_number = int(row[col_position[idx_step]])
                    if (cycle_number in self.cycle_list) and (step_number in cycles):
                        cycle_data.append([float(row[i]) for i in col_position])
                    else:
                        continue
        df = pd.DataFrame(cycle_data, columns=cols)
    
        if self.file.endswith(".csv"):
            df['time'] = df['time'] / 24 # hours to days
            df['step_time'] = df['step_time'] / 24 # hours to days
            
            # get the zeroed-capacity for each half-cycle 
            zeroed_cap = df.groupby('half_cycle').first()['capacity']
            zz_cap = [zeroed_cap[i] for i in df['half_cycle']]
            
            # capcacity data
            df['discharge_capacity'] = abs(df['capacity'] - zz_cap)*3600 # Ah to coulombs
            df['charge_capacity'] = abs(df['capacity'] - zz_cap)*3600 # Ah to coulombs
            df['capacity'] = df['capacity']*3600 # Ah to coulombs
            
            # ensure zero capacity for off-cycles
            df.loc[df['half_cycle'] % 3 != 0, 'discharge_capacity'] = 0.0
            df.loc[df['half_cycle'] % 3 == 0, 'charge_capacity'] = 0.0
            
            # re-label half_cycle numbering
            df['half_cycle'] = np.where(df['half_cycle'] % 3 == 0, 
                                        df['cycle_number']*2, (
                                            df['cycle_number']*2) - 1)
            
        else: # biologic
            df['current'] = df['current'] / 1000 # mA to amps
            df['discharge_capacity'] = df['discharge_capacity']*3.6 # mAh to coulombs
            df['charge_capacity'] = df['charge_capacity']*3.6 # mAh to coulombs
            df['capacity'] = df['capacity']*3.6 # mAh to coulombs
    
            # get the zeroed-time for each half-cycle's step_time 
            zeroed_time = df.groupby('half_cycle').first()['time']
            zz_time = [zeroed_time[i] for i in df['half_cycle']]
            
            # step_time data
            df['step_time'] = (df['time'] - zz_time) / 86400 # seconds to days
            df['time'] = df['time'] / 86400 # seconds to days
            
            # re-label half_cycle numbering
            df['half_cycle'] = np.where(df['half_cycle'] % 2 == 0, (
                df['cycle_number']*2) - 1, df['cycle_number']*2)  
            # adapt this later, no 'step' atm
            df = df[['time', 'step_time', 'voltage', 'current', 
                     'discharge_capacity', 'charge_capacity', 'capacity', 
                     'cycle_number', 'half_cycle']]
            return df

        # only novonix set up to do dqdv right now 
        df = df[['time', 'step', 'step_time', 'voltage', 'current', 
                 'discharge_capacity', 'charge_capacity', 'capacity', 
                 'cycle_number', 'half_cycle']]
        ######################################################
        # dQ/dV calculation 
        # need 2 df copies so we can filter out by threshold
        differential = df[['capacity', 'voltage']].copy()
        final_dif = df[['capacity', 'voltage']].copy()
 
        differential = differential.diff()
        differential = differential.rename(columns={"capacity": "dQ", "voltage": "dV"})

        differential['dQdV'] = differential['dQ'] / differential['dV'] 
        # check if step is a CC step (Novonix: 1,2,7, or 9). if not, voltage is NaN 
        differential['dQdV'] = np.where(df['step'].isin([1,2,7,9]) == False,
                                        np.nan, differential['dQdV'])

        differential = differential.dropna(subset=['dQdV'])

        # drops any data where dV is less than threshold
        differential = differential.drop(differential[differential.dV.abs() 
                                                      < self.dV_threshold].index)
        
        # compare back to original df to see which rows should actually be used
        compare = final_dif.index.isin(differential.index)
        # include first point again (.diff() removed first point)
        compare[0] = True
        final_dif = final_dif[compare]

        final_dif = final_dif.diff()
        final_dif = final_dif.rename(columns={"capacity": "dQ", "voltage": "dV"})
        # final dQ/dV calc now with above threshold and cleaning considered
        final_dif['dQdV'] = final_dif['dQ'] / final_dif['dV']
        final_dif = final_dif.dropna(subset=['dQdV'])
        final_dif['dQdV'] = np.where(final_dif['dQdV'] < 0, 0, final_dif['dQdV'])
        
        return df, final_dif
    #################################

    def cycle_plotter(self, step, x_axis, y_axis):

        steps = ['charge', 'discharge', 'both']
        x_options = ['time', 'step_time', 'discharge_capacity', 
                     'charge_capacity', 'capacity', 'voltage']
        y_options = ['voltage', 'current', 'discharge_capacity', 
                     'charge_capacity', 'capacity', 'dQdV']
        assert step in steps, ("Invalid step selection. Options:\
                               charge/discharge/both")
        assert x_axis in x_options, ("Invalid x-axis selection. Options: time/\
                                     step_time/discharge_capacity/charge_capacity/\
                                    capacity/voltage")
        assert y_axis in y_options, ("Invalid y-axis selection. Options: voltage/\
                                     current/discharge_capacity/charge_capacity/\
                                    capacity/dQdV")
        ##
        if self.is_novonix:
            df, fdif = self.select_cycles()
        else: # is biologic
            df = self.select_cycles()
        
        if self.plot:
            fig,ax = plt.subplots()
        else:
            x_list,y_list,v_list, dqdv_list = [],[],[],[]
    
        if step == 'charge':
            step_select = [(2*i)-1 for i in self.cycle_list]
        elif step == 'discharge':
            step_select = [2*i for i in self.cycle_list]
        else: # 'both'
            all_steps = [((2*i)-1,(2*i)) for i in self.cycle_list]
            step_select = list(sum(all_steps, ()))
        
        for cycle in self.cycle_list:
            x_data = df[x_axis].loc[(df['cycle_number'] == cycle) & (df['half_cycle'].isin(step_select))]
            y_data = df[y_axis].loc[(df['cycle_number'] == cycle) & (df['half_cycle'].isin(step_select))]
            
            if self.plot:
                ax.plot(x_data, y_data, 'o', markersize=4, label='cycle ' + str(cycle))
            else:
                x_list.append(x_data)
                y_list.append(y_data)
                
        if self.plot:
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.legend() 
            return 
        
        elif not self.is_novonix:
            return x_list, y_list
        
        else:
            # for dqdv part
            vv = df[df.index.isin(fdif.index)]

            for cycle in self.cycle_list:
                # might want to include the time too later? 
                volts = vv['voltage'].loc[(vv['cycle_number'] == cycle) & (vv['half_cycle'].isin(step_select))]
                dQdV = fdif[fdif.index.isin(volts.index)]['dQdV'] 
                v_list.append(volts.tolist())
                dqdv_list.append(dQdV.tolist())

            return x_list, y_list, v_list, dqdv_list
        

####################################
if __name__ == '__main__':
    print('testing')
