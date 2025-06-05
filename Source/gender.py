# gender
'''
Description: Functions to implement gender-specific probabilities and utilities
Authors: Myles Ingram, Elisabeth Silver
Last update: 06.14.19 (ERS)

'''


import lynch_presets as ps
import pandas as pd

class gender_obj:
    
    def __init__(self, gender_spec):
        
        self.gender = gender_spec
        if gender_spec == 'male':
            self.params = ps.params_male
            self.healthy_u = pd.read_excel(ps.params_male, "healthy_utilities", index_col=0)
            self.lynch_ac_mortality = ps.cut_ac_mortality
            
            #self.c_crc_tx, self.c_init, self.c_compl = gender_costs(self.params)
    
            #self.u_crc_tx, self.u_init = gender_util(self.params)
        elif gender_spec == 'female':
            self.params = ps.params_female
            self.healthy_u = pd.read_excel(ps.params_female, "healthy_utilities", index_col=0)
            self.lynch_ac_mortality = ps.cut_ac_mortality_fm
            
            #self.c_crc_tx, self.c_init, self.c_compl = gender_costs(self.params)
            #self.u_crc_tx, self.u_init = gender_util(self.params)
        elif gender_spec == 'both':
            self.params = ps.params_male
            self.healthy_u = pd.read_excel(ps.params_male, "healthy_utilities", index_col=0)
            self.lynch_ac_mortality = ps.cut_ac_mortality_cmb
        else:
            print("ERROR: please enter valid option (male, female, both)")
    