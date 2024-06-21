"""
Created on Fri Jul 15 08:24:44 2022

@author: Eric Fell
"""

import numpy as np
import csv


class CCCV_breakdown_novonix:
    """
    Takes input folder and Novonix cycling .csv file and extracts cycle
    capacities, timestamps, for cycling portion in CC mode and CV mode 
    
    Novonix step types:
    -on charge: when switches from 7->8 is CV, 8->9 is now discharge
    -on discharge: when switches from 9->10 is CV, 10->7 is now discharge
    
    returns:
        time_CC_charge
        CC_charge
        time_charge
        charge
        time_CC_discharge 
        CC_discharge 
        time_discharge 
        discharge
    """

    def __init__(self, folder, file):
        """Input file location of Novonix .csv file."""
        self.location = folder
        self.filename = file
        assert file.endswith('.csv'), "Not a Novonix .csv filetype"
        assert not file.endswith('_out.csv'), "Novonix file is not raw current data"

    ########################################################
    def parse_CCCV(self):
    
        col_position = [2, 3, 7] # steptype, runtime (hours), capacity
        step, runtime, capacity = col_position
        read_lines = False
        first_step = True 
        
        big_D = []
        prev_row = ['0', '0', '0']
        
        with open(self.location + '\\' + self.filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            for count, row in enumerate(csv_reader):
                # need to skip the preamble in Novonix files
                if not read_lines:
                    if 'Date and Time' in row:
                        read_lines = True
                    continue
                # gets first data point's step type. If OCV, it won't record it,
                # otherwise step is recorded        
                if first_step:
                    st = row[step]
                    if st != '0':
                        big_D.append([row[i] for i in col_position])
                    first_step = False
                        
                step_type = row[step]
                
                if step_type != st:
                    big_D.append(prev_row)
                    big_D.append([row[i] for i in col_position])
                  
                st = row[step]
                prev_row = [row[i] for i in col_position]
        # if last datapoint wasnt an end of cycle
        if big_D[-1] != prev_row:
            big_D.append(prev_row)
        # removes OCV data point
        if big_D[0][0] == '0':
            big_D.pop(0)
            
        raw_cycle_data = np.asarray(big_D, dtype=np.float64)
        return raw_cycle_data
    ##########################################################
    def cycle_capacities(self, data):
        """Returns 2D array of timestamps and absolute values of
        charge/discharge capacities.
        """
        # each complete half cycle has 4 data pts
        q, mod = divmod(len(data), 4)
        size = q*2 + (mod // 2) 
    
        c_finish = len(data)
        d_finish = len(data)
        if len(data) % 8 == 2:
            # extra charge, stopped in CC
            c_finish = -2 # slice
        elif len(data) % 8 == 6:
            # extra discharge, stopped in CC
            d_finish = -2 # ?
        else:
            pass 
        
        cap_data = np.column_stack((np.zeros(size), #data[:, 0], # time (h)
                                   np.zeros(size), # CC charge (Ah)
                                   np.zeros(size), # total charge (Ah)
                                   np.zeros(size), # CC discharge (Ah)
                                   np.zeros(size), # total discharge (Ah)
                                   ))
        # make timestamps
        cap_data[0::4, 0] = data[1::8, 1] # CC charge
        cap_data[1::4, 0] = data[3::8, 1] # total charge
        cap_data[2::4, 0] = data[5::8, 1] # CC discharge
        cap_data[3::4, 0] = data[7::8, 1] # total discharge
        
        # account for non-zero start timestamp
        cap_data[:, 0] = cap_data[:, 0] - cap_data[0, 1]
        
        # get capacities
        cap_data[0::4, 1] = abs(data[1::8, 2] - data[0::8, 2]) # CC charge
        cap_data[1::4, 2] = abs(data[3::8, 2] - data[0:c_finish:8, 2]) # total charge
        
        cap_data[2::4, 3] = abs(data[5::8, 2] - data[4::8, 2]) # CC discharge
        cap_data[3::4, 4] = abs(data[7::8, 2] - data[4:d_finish:8, 2]) # total discharge
        # dont think i need first CV step for charge/discharge..could rmv 25% data
        return cap_data
    ######################################################################
    def data_parse(self):
        raw_data = self.parse_CCCV()
        clean_data = self.cycle_capacities(raw_data)

        time_CC_charge = clean_data[0::4, 0]
        CC_charge = clean_data[0::4, 1]
        time_charge = clean_data[1::4, 0]
        charge = clean_data[1::4, 2]
        time_CC_discharge = clean_data[2::4, 0]
        CC_discharge = clean_data[2::4, 3]
        time_discharge = clean_data[3::4, 0]
        discharge = clean_data[3::4, 4]
        return (time_CC_charge, CC_charge, time_charge, charge, 
                time_CC_discharge, CC_discharge, time_discharge, discharge)   
#####################################################
if __name__ == '__main__':
    print('testing')