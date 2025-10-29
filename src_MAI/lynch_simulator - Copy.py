# lynch simulator

'''
Author: Myles Ingram
Description: Functions to run the LS markov model
Last update: 07.02.19
'''

import lynch_presets as ps
import numpy as np
import pandas as pd
import probability_functions as pf
import matplotlib.pyplot as plt
import data_manipulation as dm
#import PSA as sa
import gender as g



#utilities and costs for all arms
def unif_dist(df):
    for i in range(df.shape[1]):
        col_val = df.values[:, i]
        if col_val[0] == 0:
            continue
        else:
            val_list = []
            first_val = np.random.uniform(col_val[0]*.5, col_val[0])
            val_list.append(first_val)
            for j in range(len(col_val)-1):
                if col_val[j + 1] != col_val[j]:
                    x = np.random.uniform(col_val[j]*.5, col_val[j])
                elif j == len(col_val)-1:
                    x = np.random.uniform(col_val[j]*.5, col_val[j])
                val_list.append(x)
            df.iloc[:, i] = val_list
    return df


def gamma_dist(old_list):
    k_list, th_list = [], []        
    for x in old_list:
        k, th = pf.gamma_dist(x, x*.1)
        k_list.append(k)
        th_list.append(th)
    new_list = [np.random.gamma(k, th) for k, th in zip(k_list, th_list)]
    return new_list


def beta_dist(old_list):
    a_list, b_list = [], []
    for x in old_list:
        a, b = pf.beta_dist(x, x*.3)
        a_list.append(a)
        b_list.append(b)
    new_list = [np.random.beta(a, b) for a, b in zip(a_list, b_list)]
    return new_list


'''
FUNCTIONS TO RUN MODEL
'''

# Turns dictionaries of states and connections into a dataframe
def dict_to_connect_matrix(states, connects):
    c_matrix = np.zeros((len(states), len(states)))
    for key in connects.keys():
        for connect in connects[key]:
            if connect in ps.ALL_STATES.keys():
                c_matrix[key, connect] = 1
    return c_matrix


def switch_connect(c_matrix, row, column):
# =============================================================================
#     Switches a connection located at the given row and 
#     column on and all others in that row of connectivity matrix off 
# =============================================================================
    row_length = c_matrix.shape[-1]
    
    for connect in range(row_length):
        if connect == column:
            c_matrix[row, int(connect)] = 1
        else:
            c_matrix[row, int(connect)] = 0
        
    return c_matrix


def choose_guidelines(c_matrix, run_spec):
# =============================================================================
#     Selects which guidelines to put into the model
# =============================================================================
    if run_spec.interval == 1:
        # goes to current guidlines node
        switch_connect(c_matrix, 0, 1) 
        # negative colonoscopy goes back to current guidelines node
#        switch_connect(c_matrix, 5, 1) 
    elif run_spec.interval > 1:
        # goes to new guidlines node
        switch_connect(c_matrix, 0, 2)
        # negative colonoscopy goes back to new guidelines node
#        switch_connect(c_matrix, 5, 2)
    elif run_spec.interval == 0:
        # goes to natural history node
        switch_connect(c_matrix, 0, 3)
    else:
        print("Please enter a valid guideline: current/new/natural_history")
    return c_matrix


# initializes the start state
def get_start_state(states, run_spec):
    start_state = np.zeros((1, len(states)))
    if run_spec.interval == 0 or run_spec.start_age != ps.START_AGE:
        start_state[0][3] = 1
    elif run_spec.interval == 1:
        start_state[0][1] = 1
    else:
        start_state[0][2] = 1
    return start_state
    
# keeps 
def nonegative(array):
    new_array = []
    for element in array:
        if element <= 0:
            element = 0
        else:
            element = element
        new_array.append(element)
    return new_array


    
def create_t_matrix(run_spec, time = ps.time, age_0 = ps.START_AGE):
# =============================================================================
#     Creates a transition matrix for the model
#     Inputs are states of the model dictionary, connectivity dictionary,
#     time, and the guidelines desired as a string
# =============================================================================
    states = ps.ALL_STATES
    connect = ps.CONNECTIVITY
    
    # creates connectivity matrix
    this_gender = g.gender_obj(run_spec.gender)
    
    if run_spec.start_age != ps.START_AGE and age_0 == ps.START_AGE:
        #creates a temporary run_spec object to set adenoma transition == 0...
        #...if the start age for CSY != preset age AND we're getting 1st t_matrix
        this_run_spec = ps.run_type(0, run_spec.gene, this_gender.gender)
        
    else:
        this_run_spec = run_spec
    c_matrix = choose_guidelines(dict_to_connect_matrix(states, connect),
                                 this_run_spec)
    rand_t_matrix = np.full((len(states), len(states)), 0.0)
    
    if run_spec.interval == 0:
        csy_tracker = np.full(51, False)
    else:
        csy_tracker = np.full(51, False)
        i = 0
        for i in range(0, len(csy_tracker)):
            if i + age_0 >= run_spec.start_age:
                if i % run_spec.interval == 0:
                    csy_tracker[i] = True
                
            else:
                csy_tracker[i] = False
    # to make code more readable
    names = dm.flip(ps.ALL_STATES)

    all_cause_states = [names["current"], names["new"], names["nono"],
                        names["init adenoma"], names["adenoma"]]
    all_cause_dx_states = [names['init dx stage I'],
                           names['init dx stage II'], names['init dx stage III'],
                           names['init dx stage IV'], names['dx stage I'],
                           names['dx stage II'], names['dx stage III'],
                           names['dx stage IV']]
    csy_death_states = [names['current'], names['new'], names['init adenoma'],
                        names['adenoma']]

    
    nodes_nono, risk_probs_nono, risk_rates_nono = pf.cumul_prob_to_annual(this_gender.params, 
                                                                           this_run_spec.gene, 1)
    
    

    if type(this_run_spec.risk_ratio) != str:
        nodes, risk_probs, risk_rates = pf.cumul_prob_to_annual(this_gender.params, 
                                                                this_run_spec.gene, this_run_spec.risk_ratio)
        
        nodes_adn, risk_adn_list, adn_rates = pf.cumul_prob_KM(ps.risk_adenoma_data)
        
#        risk_probs = beta_dist(risk_probs)
   
#    risk_probs_nono = beta_dist(risk_probs_nono)
    
    
    
    for t in time:
        
        # defining risk_probs
        age = t + age_0
        if age < 60:
            colectomy_death = ps.colectomy_death_risk[0]
        elif age < 70:
            colectomy_death = ps.colectomy_death_risk[1]

        else:
            colectomy_death = ps.colectomy_death_risk[2]
#            colectomy_death = colectomy_death*.9
        
        
        if csy_tracker[t] == True:
            csy_death_states = [names['current'], names['new'], names['init adenoma'],
                                names['adenoma']]
        else:
            csy_death_states = [names['init adenoma'], names['adenoma']]
        
        #checks to see if a colonoscopy will happen this cycle
        #if type(this_run_spec.risk_ratio) != str and age >= this_run_spec.start_age:
        
        if type(this_run_spec.risk_ratio) != str:
            dx_risk_prob =  pf.pw_choose_prob(age, risk_probs, nodes)
            
            risk_adn = pf.pw_choose_prob(age, risk_adn_list, nodes_adn)
            adv_adn_risk_list = pf.annual_rate_to_prob(risk_rates*(ps.adv_adenoma_risk_mult), nodes)
            adv_adn_risk = pf.pw_choose_prob(age, adv_adn_risk_list, nodes)                        
            adn_dx_risk = pf.weighted_avg([adv_adn_risk, dx_risk_prob], 
                                              [ps.adv_adenoma_prob, 1-ps.adv_adenoma_prob])
            csy_death_risk = ps.p_csy_death

        else:
            #if no colonoscopy, adenoma risk = 0
            dx_risk_prob = pf.pw_choose_prob(age, risk_probs_nono, nodes_nono)
            adv_adn_risk = 0
            adn_dx_risk = 0
            risk_adn = 0
            csy_death_risk = 0
            
        stage_2_death, stage_3_death, stage_4_death = pf.get_cancer_death_probs(age, this_gender)
        
#        stage_2_death = stage_2_death*1.05
#        stage_2_death = stage_2_death*.95
#        stage_3_death = stage_3_death*1.05
#        stage_3_death = stage_3_death*.95
#        stage_4_death = stage_4_death*1.05
#        stage_4_death = stage_4_death*.95
        
        if t == 0:
            temp = np.multiply(rand_t_matrix, c_matrix)
            
            temp[names[this_run_spec.guidelines], names['init adenoma']] = risk_adn
#                temp[names['adenoma'], names['cancer dx']] = adn_dx_risk
            ##probabilities for adenoma -> dx
            temp[names['adenoma'], 
                 names['init dx stage I']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names['adenoma'], 
                 names['init dx stage II']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names['adenoma'], 
                 names['init dx stage III']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names['adenoma'], 
                 names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_4']
            
            temp[names['init adenoma'], 
                 names['init dx stage I']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names['init adenoma'], 
                 names['init dx stage II']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names['init adenoma'], 
                 names['init dx stage III']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names['init adenoma'], 
                 names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_4']

            
            ##probabilities for normal -> dx
            temp[names[this_run_spec.guidelines],
                 names['init dx stage I']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage II']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_2'] 
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage III']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_3'] 
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage IV']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_4'] 
            
# =============================================================================
#             if this_run_spec.guidelines != "nono":
#                 temp[names[this_run_spec.guidelines], 
#                      names["nono"]] = pf.prob_conv(1-ps.adh, 1, 1/50)
#                 temp[names["nono"],
#                  names['init dx stage I']] = dx_risk_prob * ps.staging.loc[0, 'stage_1']
#                 temp[names["nono"], 
#                  names['init dx stage II']] = dx_risk_prob * ps.staging.loc[0, 'stage_2'] 
#                 temp[names["nono"], 
#                  names['init dx stage III']] = dx_risk_prob * ps.staging.loc[0, 'stage_3'] 
#                 temp[names["nono"], 
#                  names['init dx stage IV']] = dx_risk_prob * ps.staging.loc[0, 'stage_4'] 
# =============================================================================
            
        
            
            #stage I not included since survival is 100%
            #if statements check to make sure that CRC mortality ! > all cause mortality
            temp[names['init dx stage I'], names['cancer death']] = colectomy_death
            if ps.CRC_death_rate.loc[2, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage II'], names['cancer death']] = stage_2_death
                temp[names['init dx stage II'], names['cancer death']] = stage_2_death
            if ps.CRC_death_rate.loc[3, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage III'], names['cancer death']] = stage_3_death
                temp[names['init dx stage III'], names['cancer death']] = stage_3_death
            if ps.CRC_death_rate.loc[4, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage IV'], names['cancer death']] = stage_4_death
                temp[names['init dx stage IV'], names['cancer death']] = stage_4_death
            
            for i in all_cause_states:
                temp[i, names["all cause"]] = this_gender.lynch_ac_mortality[age]
            for i in all_cause_dx_states:
                temp[i, names['all cause dx']] = this_gender.lynch_ac_mortality[age]
            for i in csy_death_states:
                temp[i, names['csy death']] = csy_death_risk
            temp[names['init adenoma'], names['adenoma']] = 1 - this_gender.lynch_ac_mortality[age] - adn_dx_risk - csy_death_risk

        else:
            temp = temp
            temp[names[this_run_spec.guidelines], names['init adenoma']] = risk_adn
#                temp[names['adenoma'], names['cancer dx']] = adn_dx_risk
            ##probabilities for adenoma -> dx
            temp[names['adenoma'], 
                 names['init dx stage I']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names['adenoma'], 
                 names['init dx stage II']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names['adenoma'], 
                 names['init dx stage III']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names['adenoma'], 
                 names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_4']
            
            temp[names['init adenoma'], 
                 names['init dx stage I']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names['init adenoma'], 
                 names['init dx stage II']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names['init adenoma'], 
                 names['init dx stage III']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names['init adenoma'], 
                 names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_4']
            
            ##probabilities for normal -> dx
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage I']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage II']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_2'] 
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage III']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage IV']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_4'] 
                 
# =============================================================================
#             if this_run_spec.guidelines != "nono":
#                 temp[names[this_run_spec.guidelines], 
#                      names["nono"]] = pf.prob_conv(1-ps.adh, 1, 1/50)
#                 temp[names["nono"],
#                  names['init dx stage I']] = dx_risk_prob * ps.staging.loc[0, 'stage_1']
#                 temp[names["nono"], 
#                  names['init dx stage II']] = dx_risk_prob * ps.staging.loc[0, 'stage_2'] 
#                 temp[names["nono"], 
#                  names['init dx stage III']] = dx_risk_prob * ps.staging.loc[0, 'stage_3'] 
#                 temp[names["nono"], 
#                  names['init dx stage IV']] = dx_risk_prob * ps.staging.loc[0, 'stage_4'] 
# =============================================================================
        
            
            temp[names['init dx stage I'], names['cancer death']] = colectomy_death
            if ps.CRC_death_rate.loc[2, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage II'], names['cancer death']] = stage_2_death
                temp[names['init dx stage II'], names['cancer death']] = stage_2_death
            if ps.CRC_death_rate.loc[3, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage III'], names['cancer death']] = stage_3_death
                temp[names['init dx stage III'], names['cancer death']] = stage_3_death
            if ps.CRC_death_rate.loc[4, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage IV'], names['cancer death']] = stage_4_death
                temp[names['init dx stage IV'], names['cancer death']] = stage_4_death
                
            for i in all_cause_states:
                temp[i, names["all cause"]] = this_gender.lynch_ac_mortality[age]
            for i in all_cause_dx_states:
                temp[i, names['all cause dx']] = this_gender.lynch_ac_mortality[age]
            for i in csy_death_states:
                temp[i, names['csy death']] = csy_death_risk
            #even if stage death > all cause, function to set probs will return 0
            temp[names['init dx stage I'], 
                 names['dx stage I']] = 1 - this_gender.lynch_ac_mortality[age] - colectomy_death
            temp[names['init dx stage II'], 
                 names['dx stage II']] = 1 - this_gender.lynch_ac_mortality[age] - stage_2_death
            temp[names['init dx stage III'], 
                 names['dx stage III']] = 1 - this_gender.lynch_ac_mortality[age] - stage_3_death
            temp[names['init dx stage IV'], 
                 names['dx stage IV']] = 1 - this_gender.lynch_ac_mortality[age] - stage_4_death
            temp[names['init adenoma'], names['adenoma']] = 1 - this_gender.lynch_ac_mortality[age] - adn_dx_risk - csy_death_risk
        
        #normalizes cells such that each cell = 1- sum(all other cells)
        for row in range(14):
            pf.normalize_new(temp[row], row)
            #print(sum(temp[row]))
        #normalizes death state rows such that all transitions other than same -> same == 0
        for row in range(14, 18):
            pf.normalize(temp[row], row)
        
        # adding depth of the matrix
        # creating the base of the 3D matrix
        if t == 0:
            t_matrix = temp
        else:
            t_matrix = np.vstack((t_matrix, temp))
    t_matrix = np.reshape(t_matrix, (len(time), len(states), len(states)))
    return t_matrix


def create_t_matrix_PSA(run_spec, time = ps.time, age_0 = ps.START_AGE):
# =============================================================================
#     Creates a transition matrix for the model
#     Inputs are states of the model dictionary, connectivity dictionary,
#     time, and the guidelines desired as a string
# =============================================================================
    states = ps.ALL_STATES
    connect = ps.CONNECTIVITY
    
    # creates connectivity matrix
    this_gender = g.gender_obj(run_spec.gender)
    
    if run_spec.start_age != ps.START_AGE and age_0 == ps.START_AGE:
        #creates a temporary run_spec object to set adenoma transition == 0...
        #...if the start age for CSY != preset age AND we're getting 1st t_matrix
        this_run_spec = ps.run_type(0, run_spec.gene, this_gender.gender)
        
    else:
        this_run_spec = run_spec
    c_matrix = choose_guidelines(dict_to_connect_matrix(states, connect),
                                 this_run_spec)
    rand_t_matrix = np.full((len(states), len(states)), 0.0)
    
    if run_spec.interval == 0:
        csy_tracker = np.full(51, False)
    else:
        csy_tracker = np.full(51, False)
        i = 0
        for i in range(0, len(csy_tracker)):
            if i + age_0 >= run_spec.start_age:
                if i % run_spec.interval == 0:
                    csy_tracker[i] = True
                
            else:
                csy_tracker[i] = False
    # to make code more readable
    names = dm.flip(ps.ALL_STATES)

    all_cause_states = [names["current"], names["new"], names["nono"],
                        names["init adenoma"], names["adenoma"]]
    all_cause_dx_states = [names['init dx stage I'],
                           names['init dx stage II'], names['init dx stage III'],
                           names['init dx stage IV'], names['dx stage I'],
                           names['dx stage II'], names['dx stage III'],
                           names['dx stage IV']]
    csy_death_states = [names['current'], names['new'], names['init adenoma'],
                        names['adenoma']]

    
    nodes_nono, risk_probs_nono, risk_rates_nono = pf.cumul_prob_to_annual(this_gender.params, 
                                                                           this_run_spec.gene, 1)
    
    

    if type(this_run_spec.risk_ratio) != str:
        nodes, risk_probs, risk_rates = pf.cumul_prob_to_annual(this_gender.params, 
                                                                this_run_spec.gene, this_run_spec.risk_ratio)
        
        nodes_adn, risk_adn_list, adn_rates = pf.cumul_prob_KM(ps.risk_adenoma_data)
        
        risk_probs = beta_dist(risk_probs)
   
    risk_probs_nono = beta_dist(risk_probs_nono)
    
    
    for t in time:
        
        # defining risk_probs
        age = t + age_0
        if age < 60:
            colectomy_death = ps.colectomy_death_risk[0]
            colectomy_death = colectomy_death*np.random.uniform(.95, 1.05)
        elif age < 70:
            colectomy_death = ps.colectomy_death_risk[1]
            colectomy_death = colectomy_death*np.random.uniform(.95, 1.05)
        else:
            colectomy_death = ps.colectomy_death_risk[2]
        colectomy_death = colectomy_death*np.random.uniform(.95, 1.05)
        
        
        if csy_tracker[t] == True:
            csy_death_states = [names['current'], names['new'], names['init adenoma'],
                                names['adenoma']]
        else:
            csy_death_states = [names['init adenoma'], names['adenoma']]
        
        #checks to see if a colonoscopy will happen this cycle
        #if type(this_run_spec.risk_ratio) != str and age >= this_run_spec.start_age:
        
        if type(this_run_spec.risk_ratio) != str:
            dx_risk_prob =  pf.pw_choose_prob(age, risk_probs, nodes)
            
            risk_adn = pf.pw_choose_prob(age, risk_adn_list, nodes_adn)
            adv_adn_risk_list = pf.annual_rate_to_prob(risk_rates*(ps.adv_adenoma_risk_mult), nodes)
            adv_adn_risk = pf.pw_choose_prob(age, adv_adn_risk_list, nodes)                        
            adn_dx_risk = pf.weighted_avg([adv_adn_risk, dx_risk_prob], 
                                              [ps.adv_adenoma_prob, 1-ps.adv_adenoma_prob])
            csy_death_risk = ps.p_csy_death

        else:
            #if no colonoscopy, adenoma risk = 0
            dx_risk_prob = pf.pw_choose_prob(age, risk_probs_nono, nodes_nono)
            adv_adn_risk = 0
            adn_dx_risk = 0
            risk_adn = 0
            csy_death_risk = 0
            
        stage_2_death, stage_3_death, stage_4_death = pf.get_cancer_death_probs(age, this_gender)
        
        stage_2_death = stage_2_death*np.random.uniform(.95, 1.05)
        stage_3_death = stage_3_death*np.random.uniform(.95, 1.05)
        stage_4_death = stage_4_death*np.random.uniform(.95, 1.05)
        
        if t == 0:
            temp = np.multiply(rand_t_matrix, c_matrix)
            
            temp[names[this_run_spec.guidelines], names['init adenoma']] = risk_adn
#                temp[names['adenoma'], names['cancer dx']] = adn_dx_risk
            ##probabilities for adenoma -> dx
            temp[names['adenoma'], 
                 names['init dx stage I']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names['adenoma'], 
                 names['init dx stage II']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names['adenoma'], 
                 names['init dx stage III']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names['adenoma'], 
                 names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_4']
            
            temp[names['init adenoma'], 
                 names['init dx stage I']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names['init adenoma'], 
                 names['init dx stage II']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names['init adenoma'], 
                 names['init dx stage III']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names['init adenoma'], 
                 names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_4']

            
            ##probabilities for normal -> dx
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage I']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage II']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage III']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage IV']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_4']
            
            #stage I not included since survival is 100%
            #if statements check to make sure that CRC mortality ! > all cause mortality
            temp[names['init dx stage I'], names['cancer death']] = colectomy_death
            if ps.CRC_death_rate.loc[2, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage II'], names['cancer death']] = stage_2_death
                temp[names['init dx stage II'], names['cancer death']] = stage_2_death
            if ps.CRC_death_rate.loc[3, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage III'], names['cancer death']] = stage_3_death
                temp[names['init dx stage III'], names['cancer death']] = stage_3_death
            if ps.CRC_death_rate.loc[4, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage IV'], names['cancer death']] = stage_4_death
                temp[names['init dx stage IV'], names['cancer death']] = stage_4_death
            
            for i in all_cause_states:
                temp[i, names["all cause"]] = this_gender.lynch_ac_mortality[age]
            for i in all_cause_dx_states:
                temp[i, names['all cause dx']] = this_gender.lynch_ac_mortality[age]
            for i in csy_death_states:
                temp[i, names['csy death']] = csy_death_risk
            temp[names['init adenoma'], names['adenoma']] = 1 - this_gender.lynch_ac_mortality[age] - adn_dx_risk - csy_death_risk

        else:
            temp = temp
            temp[names[this_run_spec.guidelines], names['init adenoma']] = risk_adn
#                temp[names['adenoma'], names['cancer dx']] = adn_dx_risk
            ##probabilities for adenoma -> dx
            temp[names['adenoma'], 
                 names['init dx stage I']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names['adenoma'], 
                 names['init dx stage II']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names['adenoma'], 
                 names['init dx stage III']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names['adenoma'], 
                 names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_4']
            
            temp[names['init adenoma'], 
                 names['init dx stage I']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names['init adenoma'], 
                 names['init dx stage II']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names['init adenoma'], 
                 names['init dx stage III']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names['init adenoma'], 
                 names['init dx stage IV']] = adn_dx_risk * ps.staging.loc[this_run_spec.interval, 'stage_4']
            
            ##probabilities for normal -> dx
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage I']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_1']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage II']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_2']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage III']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_3']
            temp[names[this_run_spec.guidelines], 
                 names['init dx stage IV']] = dx_risk_prob * ps.staging.loc[this_run_spec.interval, 'stage_4']
            
            temp[names['init dx stage I'], names['cancer death']] = colectomy_death
            if ps.CRC_death_rate.loc[2, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage II'], names['cancer death']] = stage_2_death
                temp[names['init dx stage II'], names['cancer death']] = stage_2_death
            if ps.CRC_death_rate.loc[3, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage III'], names['cancer death']] = stage_3_death
                temp[names['init dx stage III'], names['cancer death']] = stage_3_death
            if ps.CRC_death_rate.loc[4, 'death_rate'] > this_gender.lynch_ac_mortality[age]:
                temp[names['dx stage IV'], names['cancer death']] = stage_4_death
                temp[names['init dx stage IV'], names['cancer death']] = stage_4_death
                
            for i in all_cause_states:
                temp[i, names["all cause"]] = this_gender.lynch_ac_mortality[age]
            for i in all_cause_dx_states:
                temp[i, names['all cause dx']] = this_gender.lynch_ac_mortality[age]
            for i in csy_death_states:
                temp[i, names['csy death']] = csy_death_risk
            #even if stage death > all cause, function to set probs will return 0
            temp[names['init dx stage I'], 
                 names['dx stage I']] = 1 - this_gender.lynch_ac_mortality[age] - colectomy_death
            temp[names['init dx stage II'], 
                 names['dx stage II']] = 1 - this_gender.lynch_ac_mortality[age] - stage_2_death
            temp[names['init dx stage III'], 
                 names['dx stage III']] = 1 - this_gender.lynch_ac_mortality[age] - stage_3_death
            temp[names['init dx stage IV'], 
                 names['dx stage IV']] = 1 - this_gender.lynch_ac_mortality[age] - stage_4_death
            temp[names['init adenoma'], names['adenoma']] = 1 - this_gender.lynch_ac_mortality[age] - adn_dx_risk - csy_death_risk
        
        #normalizes cells such that each cell = 1- sum(all other cells)
        for row in range(14):
            pf.normalize_new(temp[row], row)
#            print(sum(temp[row]))
            #print(sum(temp[row]))
        #normalizes death state rows such that all transitions other than same -> same == 0
        for row in range(14, 18):
            pf.normalize(temp[row], row)
        
        # adding depth of the matrix
        # creating the base of the 3D matrix
        if t == 0:
            t_matrix = temp
        else:
            t_matrix = np.vstack((t_matrix, temp))
    t_matrix = np.reshape(t_matrix, (len(time), len(states), len(states)))
    return t_matrix


def av_matrix(run_spec):
    age_start = ps.START_AGE
    
    if run_spec.start_age == ps.START_AGE:
        #if CSY age = start age, run model w/o age vary
        t_matrix = create_t_matrix(run_spec)
    else:
        #get the number of years w/o CSY
        end_age_1 = run_spec.start_age - age_start
        #get the number of years w/ CSY
        end_age_2 = ps.NUM_YEARS - end_age_1
        #create ranges to bound t_matrices
        time_1 = range(end_age_1)
        time_2 = range(end_age_2)
        t_matrix_1 = create_t_matrix(run_spec, time = time_1, age_0 = age_start)
        t_matrix_2 = create_t_matrix(run_spec, time = time_2, age_0 = run_spec.start_age)
        t_matrix = np.vstack((t_matrix_1, t_matrix_2))
        #csy_checker = np.append(csy_checker_1, csy_checker_2)
        
    return t_matrix


def av_matrix_PSA(run_spec):
    age_start = ps.START_AGE
    
    if run_spec.start_age == ps.START_AGE:
        #if CSY age = start age, run model w/o age vary
        t_matrix = create_t_matrix(run_spec)
    else:
        #get the number of years w/o CSY
        end_age_1 = run_spec.start_age - age_start
        #get the number of years w/ CSY
        end_age_2 = ps.NUM_YEARS - end_age_1
        #create ranges to bound t_matrices
        time_1 = range(end_age_1)
        time_2 = range(end_age_2)
        t_matrix_1 = create_t_matrix_PSA(run_spec, time = time_1, age_0 = age_start)
        t_matrix_2 = create_t_matrix_PSA(run_spec, time = time_2, age_0 = run_spec.start_age)
        t_matrix = np.vstack((t_matrix_1, t_matrix_2))
        #csy_checker = np.append(csy_checker_1, csy_checker_2)
        
    return t_matrix


def run_markov_simple(run_spec):
#    print('In run_markov_simple..')
    states = ps.ALL_STATES
    age_0 = ps.START_AGE
    
    
    time = ps.time
    t_matrix = av_matrix(run_spec)
#    print(t_matrix)
    
    #if CSY start age != 25, start state is "nono", then flips to guidelines in for loop
    start_state = get_start_state(states, run_spec)
    # creates a DataFrame for the population in each state for each time point
    D_matrix = pd.DataFrame(start_state, columns=states)
    cancer_incidence = np.zeros(51)
    overall_survival = np.zeros(51)
    overall_survival[0] = 1
    age = np.zeros(51)
    age[0] = ps.START_AGE
    if run_spec.interval == 0:
        csy_tracker = np.full(51, False)
    else:
        csy_tracker = np.full(51, False)
        i = 0
        for i in range(0, len(csy_tracker)):
            if i + age_0 >= run_spec.start_age:
                if i % run_spec.interval == 0:
                    csy_tracker[i] = True
                
            else:
                csy_tracker[i] = False
    
    names = dm.flip(states)
    
    cancer_states = [names['init dx stage I'], names['init dx stage II'], 
                     names['init dx stage III'], names['init dx stage IV'], 
                     names['dx stage I'], names['dx stage II'], 
                     names['dx stage III'], names['dx stage IV'],
                     names['all cause dx'], names['cancer death']]
    death_states = [names['all cause dx'], names['cancer death'],
                    names['all cause'], names['csy death']]
    
    # creates population distribution at time t
    for t in time:
        if t == 0:
            Distribution = start_state
            
        elif t == run_spec.start_age - age_0:
            #print(Distribution)
            temp = Distribution
            Distribution[names[run_spec.guidelines]] = temp[3]
            Distribution[3] = 0
            
        temp = np.transpose(t_matrix[t]) * Distribution
        Distribution = [sum(temp[i, :]) for i in range(len(states))]
        
        
        age[t + 1] = age[0] + t + 1
        death_temp = 0
        for i in death_states:
            death_temp += Distribution[i]
        
        overall_survival[t + 1] = 1 - death_temp
        
        for i in cancer_states:
            cancer_incidence[t + 1] += Distribution[i]
        D_matrix.loc[len(D_matrix)] = Distribution
        
    
# =============================================================================
#         print('-'*30)
#         print('CSY Q%dY'%run_spec.interval)
#         print('Gene:', run_spec.gene)
#         print('CSY start age:', run_spec.start_age)
#         print('age:', age_0+t)
#         print('cancer incidence:', np.sum(cancer_incidence[t]))
#         print('num colonoscopies:', csy_tracker)
#         print('-'*30)
# =============================================================================
        
    #print(D_matrix)
    D_matrix.columns = list(ps.ALL_STATES.values())
    
    D_matrix['cancer incidence'] = cancer_incidence
    D_matrix['overall survival'] = overall_survival
    D_matrix['age'] = age
    D_matrix['csy tracker'] = csy_tracker
    return D_matrix, t_matrix
        

def run_markov_PSA(run_spec):
#    print('In run_markov_simple..')
    states = ps.ALL_STATES
    age_0 = ps.START_AGE
    
    
    time = ps.time
    t_matrix = av_matrix_PSA(run_spec)
#    print(t_matrix)
    
    #if CSY start age != 25, start state is "nono", then flips to guidelines in for loop
    start_state = get_start_state(states, run_spec)
    # creates a DataFrame for the population in each state for each time point
    D_matrix = pd.DataFrame(start_state, columns=states)
    cancer_incidence = np.zeros(51)
    overall_survival = np.zeros(51)
    overall_survival[0] = 1
    age = np.zeros(51)
    age[0] = ps.START_AGE
    if run_spec.interval == 0:
        csy_tracker = np.full(51, False)
    else:
        csy_tracker = np.full(51, False)
        i = 0
        for i in range(0, len(csy_tracker)):
            if i + age_0 >= run_spec.start_age:
                if i % run_spec.interval == 0:
                    csy_tracker[i] = True
                
            else:
                csy_tracker[i] = False
    
    names = dm.flip(states)
    
    cancer_states = [names['init dx stage I'], names['init dx stage II'], 
                     names['init dx stage III'], names['init dx stage IV'], 
                     names['dx stage I'], names['dx stage II'], 
                     names['dx stage III'], names['dx stage IV'],
                     names['all cause dx'], names['cancer death']]
    death_states = [names['all cause dx'], names['cancer death'],
                    names['all cause'], names['csy death']]
    
    # creates population distribution at time t
    for t in time:
        if t == 0:
            Distribution = start_state
            
        elif t == run_spec.start_age - age_0:
            #print(Distribution)
            temp = Distribution
            Distribution[names[run_spec.guidelines]] = temp[3]
            Distribution[3] = 0
            
        temp = np.transpose(t_matrix[t]) * Distribution
        Distribution = [sum(temp[i, :]) for i in range(len(states))]
        
        
        age[t + 1] = age[0] + t + 1
        death_temp = 0
        for i in death_states:
            death_temp += Distribution[i]
        
        overall_survival[t + 1] = 1 - death_temp
        
        for i in cancer_states:
            cancer_incidence[t + 1] += Distribution[i]
        D_matrix.loc[len(D_matrix)] = Distribution
        
    
# =============================================================================
#         print('-'*30)
#         print('CSY Q%dY'%run_spec.interval)
#         print('Gene:', run_spec.gene)
#         print('CSY start age:', run_spec.start_age)
#         print('age:', age_0+t)
#         print('cancer incidence:', np.sum(cancer_incidence[t]))
#         print('num colonoscopies:', csy_tracker)
#         print('-'*30)
# =============================================================================
        
    #print(D_matrix)
    D_matrix.columns = list(ps.ALL_STATES.values())
    
    D_matrix['cancer incidence'] = cancer_incidence
    D_matrix['overall survival'] = overall_survival
    D_matrix['age'] = age
    D_matrix['csy tracker'] = csy_tracker
    return D_matrix, t_matrix
# =============================================================================
# run_spec = ps.run_type(0, 'MLH1', 'female')
# d_mat, t_mat = run_markov_simple(run_spec)
# d_mat.to_csv('test_output_f.csv')
# 
# =============================================================================

def owsa_t_matrix(t_matrix, s_1, s_2, multiplier):
    '''
    For a given t_matrix gives the transition probability of going from 
    state_1 to state_2 times a multiplier
    '''
    # dimensions of the t_matrix
    depth = t_matrix.shape[0]
    length = t_matrix.shape[2]
    width = t_matrix.shape[1]
    
    # flips the number and names of the states to code more readable
    names = dm.flip(ps.ALL_STATES)
#    print(s_1)
#    print(names[s_2])
#    print(t_matrix[:, names[s_1], names[s_2]]*multiplier)
#    print(t_matrix[:, names[s_1], names[s_2]])
    t_matrix[:, names[s_1], names[s_2]] = t_matrix[:, names[s_1], names[s_2]]*multiplier
    for stack in range(depth):
#        print(depth)
        pf.normalize_new(t_matrix[stack, names[s_1], :], names[s_2])
    return t_matrix
    
    


def run_markov_OWSA(run_spec, state_1, state_2, multiplier):
#    print('In run_markov_simple..')
    states = ps.ALL_STATES
    age_0 = ps.START_AGE
    
    names = dm.flip(states)
    
    time = ps.time
    t_matrix = av_matrix(run_spec)
    t_matrix = owsa_t_matrix(t_matrix, state_1, state_2, multiplier)
#    print(t_matrix)
    
    
    
    #if CSY start age != 25, start state is "nono", then flips to guidelines in for loop
    start_state = get_start_state(states, run_spec)
    # creates a DataFrame for the population in each state for each time point
    D_matrix = pd.DataFrame(start_state, columns=states)
    cancer_incidence = np.zeros(51)
    overall_survival = np.zeros(51)
    overall_survival[0] = 1
    age = np.zeros(51)
    age[0] = ps.START_AGE
    if run_spec.interval == 0:
        csy_tracker = np.full(51, False)
    else:
        csy_tracker = np.full(51, False)
        i = 0
        for i in range(0, len(csy_tracker)):
            if i + age_0 >= run_spec.start_age:
                if i % run_spec.interval == 0:
                    csy_tracker[i] = True
                
            else:
                csy_tracker[i] = False
    
    
    
    cancer_states = [names['init dx stage I'], names['init dx stage II'], 
                     names['init dx stage III'], names['init dx stage IV'], 
                     names['dx stage I'], names['dx stage II'], 
                     names['dx stage III'], names['dx stage IV'],
                     names['all cause dx'], names['cancer death']]
    death_states = [names['all cause dx'], names['cancer death'],
                    names['all cause'], names['csy death']]
    
    # creates population distribution at time t
    for t in time:
        if t == 0:
            Distribution = start_state
            
        elif t == run_spec.start_age - age_0:
            #print(Distribution)
            temp = Distribution
            Distribution[names[run_spec.guidelines]] = temp[3]
            Distribution[3] = 0
            
        temp = np.transpose(t_matrix[t]) * Distribution
        Distribution = [sum(temp[i, :]) for i in range(len(states))]
        
        
        age[t + 1] = age[0] + t + 1
        death_temp = 0
        for i in death_states:
            death_temp += Distribution[i]
        
        overall_survival[t + 1] = 1 - death_temp
        
        for i in cancer_states:
            cancer_incidence[t + 1] += Distribution[i]
        D_matrix.loc[len(D_matrix)] = Distribution
        
    
# =============================================================================
#         print('-'*30)
#         print('CSY Q%dY'%run_spec.interval)
#         print('Gene:', run_spec.gene)
#         print('CSY start age:', run_spec.start_age)
#         print('age:', age_0+t)
#         print('cancer incidence:', np.sum(cancer_incidence[t]))
#         print('num colonoscopies:', csy_tracker)
#         print('-'*30)
# =============================================================================
        
    #print(D_matrix)
    D_matrix.columns = list(ps.ALL_STATES.values())
    
    D_matrix['cancer incidence'] = cancer_incidence
    D_matrix['overall survival'] = overall_survival
    D_matrix['age'] = age
    D_matrix['csy tracker'] = csy_tracker
    return D_matrix, t_matrix

'''

FUNCTIONS FOR GRAPHING RESULTS, GENERATING .PNGs, .CSVs

Note: all of these functions have an overwrite_file keyword. If set to True, the function
will overwrite a file with the same name (e.g., if file was already generated but there are bug fixes)

If overwrite_file is not set to True, then a new file will be created only if a file with
the same file name does not already exist (can also prompt a new file by changing the filename)

'''

'''
Graphing by colonoscopy interval:
'''

#Generates incidence graphs for specified parameters. Reverts to presets if none specified
def CRC_results_by_interval(genes = ps.genes, genders = ps.genders, 
                            intervals = ps.intervals, csy_start_age = ps.START_AGE,
                            overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)
        for gene in genes:
            cancer_df = pd.DataFrame()
            for i in intervals:
                run_spec = ps.run_type(i, gene, this_gender.gender, age_spec = csy_start_age)
            
                D_matrix, t_matrix = run_markov_simple(run_spec)
                
                cancer_df[run_spec.interval_str] = D_matrix.loc[1:, 'cancer incidence']
                
            plt.figure()
            plt.title('CRC Incidence, '+ gene + ', '+ 'Start age: '+ str(csy_start_age))
            plt.xlabel('Age')
            plt.ylabel('Incidence')
            cols = cancer_df.columns
            for col in cols:
                plt.plot(ps.age_time, cancer_df[col], label = col)
                
            #sp_age, sp_risk = dm.excel_to_lists(this_gender.params, "Sporadic")
            #plt.plot(sp_age, np.divide(sp_risk, 100), 'k-*', label= "Sporadic")
            plt.legend()
            print(cancer_df)
            ymax = cancer_df['Q0Y'].max() +.1
            plt.axis([25, 75, 0, ymax])
            file_name = 'CRC_incidence_' + gene +'_age_constant_' +str(csy_start_age)+'_combined_gender.png'
            
            if pf.check_valid_file(file_name) == False or overwrite_file == True:
                plt.savefig(ps.crc_results/file_name, dpi = 200)
            plt.show()
            
            
    return()
    


#CRC_results_by_interval(genes = ['MLH1', 'PMS2'], intervals = [0, 3])

#Generates OS graphs for specified parameters. Reverts to presets if none specified
def OS_results_by_interval(genes = ps.genes, genders = ps.genders, 
                            intervals = ps.intervals, csy_start_age = ps.START_AGE,
                            overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)
        for gene in genes:
            os_df = pd.DataFrame()
            for i in intervals:
                run_spec = ps.run_type(i, gene, this_gender.gender, 
                                       age_spec = csy_start_age)
            #this_gender = g.gender_obj(run_spec.gender)
                D_matrix, t_matrix = run_markov_simple(run_spec)
                
                os_df[run_spec.interval_str] = D_matrix.loc[1:, 'overall survival']
            
            plot_label = 'Overall Survival, '+ gene + ', '+ 'Start age: '+ str(csy_start_age)
            plt.figure()
            plt.title(plot_label)
            plt.xlabel('Age')
            plt.ylabel('% Alive')
            cols = os_df.columns
            for col in cols:
                plt.plot(ps.age_time, os_df[col], label = col)

            plt.legend()
            file_name = 'OS_' + gene +'_age_constant_' +str(csy_start_age)+'_combined_gender.png'
            plt.axis([25, 75, 0, 1])
            if pf.check_valid_file(file_name) == False or overwrite_file == True:
                plt.savefig(ps.os_results/file_name, dpi = 200)
            plt.show()
    return()
            
#OS_results_by_interval(genes = ['MLH1', 'PMS2'], intervals = [1, 2])

#Generates Cancer Death graphs for specified parameters. Reverts to presets if none specified
def CD_results_by_interval(genes = ps.genes, genders = ps.genders, 
                            intervals = ps.intervals, csy_start_age = ps.START_AGE,
                            overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)
        for gene in genes:
            cd_df = pd.DataFrame()
            for i in intervals:
                run_spec = ps.run_type(i, gene, this_gender.gender, 
                                       age_spec = csy_start_age)
            #this_gender = g.gender_obj(run_spec.gender)
                D_matrix, t_matrix = run_markov_simple(run_spec)
                
                cd_df[run_spec.interval_str] = D_matrix.loc[1:, 'cancer death']
            
            
            plt.figure()
            plt.title('Cancer-Specific Death, '+ gene + ', '+ 'Start age: '+ str(csy_start_age))
            plt.xlabel('Age')
            plt.ylabel('Cancer Mortality')
            cols = cd_df.columns
            for col in cols:
                plt.plot(ps.age_time, cd_df[col], label = col)
            file_name = 'cancer_death_' + gene +'_age_constant_'+str(csy_start_age)+'_combined_gender.png'
            plt.legend()
            ymax = cd_df['Q0Y'].max() +.1
            plt.axis([25, 75, 0, ymax])
            if pf.check_valid_file(file_name) == False or overwrite_file == True:
                plt.savefig(ps.cd_results/file_name, dpi = 200)
            plt.show()
    return

def AC_results_by_interval(genes = ps.genes, genders = ps.genders, 
                            intervals = ps.intervals, csy_start_age = ps.START_AGE,
                            overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)
        for gene in genes:
            ac_df = pd.DataFrame()
            for i in intervals:
                run_spec = ps.run_type(i, gene, this_gender.gender, 
                                       age_spec = csy_start_age)
            #this_gender = g.gender_obj(run_spec.gender)
                D_matrix, t_matrix = run_markov_simple(run_spec)
                
                D_temp = D_matrix.iloc[1:, :]
                ac_df[run_spec.interval_str] = D_temp['all cause dx'] + D_temp['all cause']
            
            
            plt.figure()
            plt.title('All cause mortality, '+ gene + ', '+ 'Start age: '+ str(csy_start_age))
            plt.xlabel('Age')
            plt.ylabel('% Dead')
            cols = ac_df.columns
            for col in cols:
                plt.plot(ps.age_time, ac_df[col], label = col)
            file_name = 'ac_death_' + gene +'_age_constant_' +str(csy_start_age)+'_combined_gender.png'
            plt.legend()
            plt.axis([25, 75, 0, 1])
            if pf.check_valid_file(file_name) == False or overwrite_file == True:
                plt.savefig(file_name, dpi = 200)
            plt.show()
    return
            
#AC_results_by_interval(intervals = [0, 1, 2, 3, 4, 5])


def graph_everything_by_int(genes_every = ps.genes, genders_every = ps.genders, 
                            intervals_every = ps.intervals,  
                            start_ages = [ps.START_AGE], overwrite_pngs = False):
    
    for i in start_ages:
        
        CRC_results_by_interval(genes = genes_every, genders = genders_every, 
                                intervals = intervals_every, csy_start_age = i,
                                overwrite_file = overwrite_pngs)
        OS_results_by_interval(genes = genes_every, genders = genders_every, 
                                intervals = intervals_every, csy_start_age = i,
                                overwrite_file = overwrite_pngs)
        CD_results_by_interval(genes = genes_every, genders = genders_every, 
                                intervals = intervals_every, csy_start_age = i,
                                overwrite_file = overwrite_pngs)
        AC_results_by_interval(genes = ps.genes, genders = ps.genders, 
                               intervals = ps.intervals, csy_start_age = i,
                                overwrite_file = overwrite_pngs)
    

#graph_everything_by_int(intervals_every = [0, 1, 2, 3, 4, 5], overwrite_pngs = True)
#graph_everything_by_int(intervals_every = [0, 1, 2, 3, 4, 5], start_ages = [25, 30], overwrite_pngs = False)


'''
Graphing by colonoscopy start age:
'''

def strat_line_style(strat):
    '''
    Selects color and shape for given strategy
    '''
    # selects color based on colonoscopy frequency
    if "Q1Y" in strat:
        color = "r"
    elif "Q2Y" in strat:
        color = "m"
    elif "Q3Y" in strat:
        color = "g"
    elif "Q4Y" in strat:
        color = "b"
    elif "Q5Y" in strat:
        color = "c"        
    
    # selccts marker based on start age
    if "30" in strat:
        mark = "1"
    elif "35" in strat:
        mark = "."
    elif "40" in strat:
        mark = "--"
    elif "45" in strat:
        mark = "2"
    elif "50" in strat:
        mark = "+"
    else:
        mark = ""
        
    style = color + mark
    
    return style

def CD_optimal_only():
    genes = ps.genes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 12))
    ax_array = [ax1, ax2, ax3, ax4]
    plt.suptitle('Colon Cancer Related Death by Gene',
                 fontsize = 14, y = 0.94)
    i = 0
    for gene in genes:
        cancer_df = pd.DataFrame()
        these_strats = ps.BC_BK_STRATS[gene]
        k = 0
        for k in range(0, len(these_strats)):
            run_info = these_strats[k]
            interval = run_info[0]
            age_start = run_info[1]
            print(interval, age_start)
            run_spec = ps.run_type(interval, gene, ps.genders[0], age_spec = age_start)
            D_matrix, t_mat = run_markov_simple(run_spec)
            cancer_df[run_spec.label] = D_matrix.loc[1:, 'cancer death']
            if k == 0:
                least_agro = run_spec.label
        ax_array[i].set_title(gene)
        ax_array[i].set_ylabel('Cancer-Related Death')
        ax_array[i].set_xlabel('Age')
            #plt.xlabel('Age')
            #plt.ylabel('CRC Incidence')
        #print(cancer_df.index.values)
        cols = cancer_df.columns.values
        for col in cols:
            style = strat_line_style(col[2:])
            ax_array[i].plot(ps.age_time, cancer_df[col], style, label = col[2:])
        ax_array[i].legend()
        ymax = cancer_df[least_agro].max() + 0.01
            #file_name = 'CRC_incidence_'+ gene + '_'+ 'interval_constant_'+ str(interval) + '_combined_gender.png'
        ax_array[i].set_ylim(bottom = 0, top = ymax)
        ax_array[i].set_xlim(left = 25, right = 75)
        i += 1
    plt.savefig(ps.cd_results/'optimal_strats_cd.png', bbox_inches = 'tight', dpi = 500)


def CD_natural_history():
    genes = ps.genes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 12))
    ax_array = [ax1, ax2, ax3, ax4]
    plt.suptitle('Natural History Colon Cancer Related Death by Gene',
                 fontsize = 14, y = 0.94)
    i = 0
    for gene in genes:
        cancer_df = pd.DataFrame()
        these_strats = ps.NAT_HIST[gene]
        k = 0
        for k in range(0, len(these_strats)):
            run_info = these_strats[k]
            interval = run_info[0]
            age_start = run_info[1]
            print(interval, age_start)
            run_spec = ps.run_type(interval, gene, ps.genders[0], age_spec = age_start)
            D_matrix, t_mat = run_markov_simple(run_spec)
            cancer_df[run_spec.label] = D_matrix.loc[1:, 'cancer death']
            if k == 0:
                least_agro = run_spec.label
        ax_array[i].set_title(gene)
        ax_array[i].set_ylabel('Cancer-Related Death')
        ax_array[i].set_xlabel('Age')
            #plt.xlabel('Age')
            #plt.ylabel('CRC Incidence')
        #print(cancer_df.index.values)
        cols = cancer_df.columns.values
        for col in cols:
            ax_array[i].plot(ps.age_time, cancer_df[col], label = "Natural History")
        ax_array[i].legend()
        ymax = cancer_df[least_agro].max() + 0.01
            #file_name = 'CRC_incidence_'+ gene + '_'+ 'interval_constant_'+ str(interval) + '_combined_gender.png'
        ax_array[i].set_ylim(bottom = 0, top = ymax)
        ax_array[i].set_xlim(left = 25, right = 75)
        i += 1
    plt.savefig(ps.cd_results/'optimal_strats_nat_hist.png', bbox_inches = 'tight', dpi = 500)



def CRC_optimal_only():
    genes = ps.genes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 12))
    ax_array = [ax1, ax2, ax3, ax4]
    plt.suptitle('Colon Cancer Incidence by Gene',
                 fontsize = 14, y = 0.94)
    i = 0
    for gene in genes:
        cancer_df = pd.DataFrame()
        these_strats = ps.BC_BK_STRATS[gene]
        k = 0
        for k in range(0, len(these_strats)):
            run_info = these_strats[k]
            interval = run_info[0]
            age_start = run_info[1]
            print(interval, age_start)
            run_spec = ps.run_type(interval, gene, ps.genders[0], age_spec = age_start)
            D_matrix, t_mat = run_markov_simple(run_spec)
            cancer_df[run_spec.label] = D_matrix.loc[1:, 'cancer incidence']
            if k == 0:
                least_agro = run_spec.label
        ax_array[i].set_title(gene)
        ax_array[i].set_ylabel('Cancer Incidence')
        ax_array[i].set_xlabel('Age')
            #plt.xlabel('Age')
            #plt.ylabel('CRC Incidence')
        #print(cancer_df.index.values)
        cols = cancer_df.columns.values
        for col in cols:
            style = strat_line_style(col[2:])
            ax_array[i].plot(ps.age_time, cancer_df[col], style, label = col[2:])
        ax_array[i].legend()
        ymax = cancer_df[least_agro].max() + 0.01
            #file_name = 'CRC_incidence_'+ gene + '_'+ 'interval_constant_'+ str(interval) + '_combined_gender.png'
        ax_array[i].set_ylim(bottom = 0, top = ymax)
        ax_array[i].set_xlim(left = 25, right = 75)
        i += 1
    plt.savefig(ps.crc_results/'optimal_strats_crc.png', bbox_inches = 'tight', dpi = 500)
    return


def CRC_natural_history():
    genes = ps.genes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 12))
    ax_array = [ax1, ax2, ax3, ax4]
    plt.suptitle('Natural History Colon Cancer Incidence by Gene',
                 fontsize = 14, y = 0.94)
    i = 0
    for gene in genes:
        cancer_df = pd.DataFrame()
        these_strats = ps.NAT_HIST[gene]
        k = 0
        for k in range(0, len(these_strats)):
            run_info = these_strats[k]
            interval = run_info[0]
            age_start = run_info[1]
            print(interval, age_start)
            run_spec = ps.run_type(interval, gene, ps.genders[0], age_spec = age_start)
            D_matrix, t_mat = run_markov_simple(run_spec)
            cancer_df[run_spec.label] = D_matrix.loc[1:, 'cancer incidence']
            if k == 0:
                least_agro = run_spec.label
        ax_array[i].set_title(gene)
        ax_array[i].set_ylabel('Cancer Incidence')
        ax_array[i].set_xlabel('Age')
            #plt.xlabel('Age')
            #plt.ylabel('CRC Incidence')
        #print(cancer_df.index.values)
        cols = cancer_df.columns.values
        for col in cols:
            ax_array[i].plot(ps.age_time, cancer_df[col], label = "Natural History")
        ax_array[i].legend()
        ymax = cancer_df[least_agro].max() + 0.01
            #file_name = 'CRC_incidence_'+ gene + '_'+ 'interval_constant_'+ str(interval) + '_combined_gender.png'
        ax_array[i].set_ylim(bottom = 0, top = ymax)
        ax_array[i].set_xlim(left = 25, right = 75)
        i += 1
    plt.savefig(ps.crc_results/'optimal_strats_nat_hist.png', bbox_inches = 'tight', dpi = 500)
    
    
#same as above, but with different starting ages. default interval is 1 year
def CRC_results_by_age(genes = ps.genes, genders = ps.genders, 
                       interval = 0, csy_start_ages = ps.ages,
                       overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)
        cancer_df = pd.DataFrame()
        for gene in genes:
            run_spec = ps.run_type(interval, gene, this_gender.gender, 
                                       age_spec = 25)
            #this_gender = g.gender_obj(run_spec.gender)
            D_matrix, t_matrix = run_markov_simple(run_spec)
                
            cancer_df[str(run_spec.gene)] = D_matrix.loc[1:, 'cancer death']
            
        plt.figure()
        plt.title('Natural History CRC-Related Death by Gene') 
        plt.xlabel('Age')
        plt.ylabel('Cancer-Related Death')
        cols = cancer_df.columns
        for col in cols:
            plt.plot(ps.age_time, cancer_df[col], label = col)
            #sp_age, sp_risk = dm.excel_to_lists(this_gender.params, "Sporadic")
            #plt.plot(sp_age, np.divide(sp_risk, 100), 'k-*', label= "Sporadic")
        plt.legend()
            
        ymax = cancer_df['40'].max() +.1
        file_name = 'CRC_incidence_'+ gene + '_'+ 'interval_constant_'+ str(interval) + '_combined_gender.png'
        plt.axis([25, 75, 0, ymax])
        if pf.check_valid_file(file_name) == False or overwrite_file == True:
            plt.savefig(ps.crc_results/file_name, dpi = 200)
            
        plt.show()
    return

def CRC_all_strats():
    '''
    creates CRC graphs for all strategies
    '''
    for intrvl in ps.intervals:
        CRC_results_by_age(interval = intrvl)
    return
        


#same as above, but with different starting ages. default interval is 1 year
def CD_results_by_age(genes = ps.genes, genders = ps.genders, 
                      interval = 3, csy_start_ages = ps.ages,
                      overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)
        for gene in genes:
            cd_df = pd.DataFrame()
            for i in csy_start_ages:
                run_spec = ps.run_type(interval, gene, this_gender.gender, 
                                       age_spec = i)
            #this_gender = g.gender_obj(run_spec.gender)
                D_matrix, t_matrix = run_markov_simple(run_spec)
                
                cd_df[str(run_spec.start_age)] = D_matrix.loc[1:, 'cancer death']
            
            
            plt.figure()
            plt.title('Cancer-Specific Death, '+ gene + ', '+ 'Interval: '+ str(interval))
            plt.xlabel('Age')
            plt.ylabel('Cancer Mortality')
            cols = cd_df.columns
            for col in cols:
                plt.plot(ps.age_time, cd_df[col], label = col)
            if gene == 'PMS2' or gene == 'MSH6':
                ymax = 0.15
            else:
                ymax = 0.3
            
            plt.legend()
            ymax = cd_df['40'].max() + .1
            plt.axis([25, 75, 0, ymax])
            file_name = 'cancer_death_'+ gene + '_'+ 'interval_constant_'+ str(interval) + '_combined_gender.png'
            if pf.check_valid_file(file_name) == False or overwrite_file == True:
                plt.savefig(ps.cd_results/file_name, dpi = 200)
            plt.show()
    return


def CD_all_strats():
    '''
    creates CD graphs for all strats
    '''
    for intrvl in ps.intervals:
        CD_results_by_age(interval=intrvl)
    return
#CD_results_by_age(genes = ['MLH1'], csy_start_ages = [25, 30])

def OS_results_by_age(genes = ps.genes, genders = ps.genders, 
                      interval = 1, csy_start_ages = ps.ages,
                      overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)
        for gene in genes:
            os_df = pd.DataFrame()
            for i in csy_start_ages:
                run_spec = ps.run_type(interval, gene, this_gender.gender, 
                                       age_spec = i)
                D_matrix, t_matrix = run_markov_simple(run_spec)
                
                os_df[str(run_spec.start_age)] = D_matrix.loc[1:, 'overall survival']
            
            
            plt.figure()
            plt.title('Overall Survival, '+ gene + ', '+ 'Interval: '+ str(interval))
            plt.xlabel('Age')
            plt.ylabel('% Alive')
            cols = os_df.columns
            for col in cols:
                plt.plot(ps.age_time, os_df[col], label = col)

            plt.legend()
            
            plt.axis([25, 75, 0, 1])
            file_name = 'overall_survival_'+ gene + '_'+ 'interval_constant_'+ str(interval) + '_combined_gender.png'
            if pf.check_valid_file(file_name) == False or overwrite_file == True:
                plt.savefig(ps.os_results/file_name, dpi = 200)
            plt.show()
    return


def OS_all_strats():
    
    for intrvl in ps.intervals:
        OS_results_by_age(interval=intrvl)
    return

def AC_results_by_age(genes = ps.genes, genders = ps.genders, 
                      interval = 1, csy_start_ages = ps.ages,
                      overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)
        for gene in genes:
            ac_df = pd.DataFrame()
            for i in csy_start_ages:
                run_spec = ps.run_type(interval, gene, this_gender.gender, 
                                       age_spec = i)
            #this_gender = g.gender_obj(run_spec.gender)
                D_matrix, t_matrix = run_markov_simple(run_spec)
                
                D_temp = D_matrix.iloc[1:, :]
                ac_df[str(run_spec.start_age)] = D_temp['all cause dx'] + D_temp['all cause']
            
            
            plt.figure()
            plt.title('All cause mortality, '+ gene + ', '+ 'Interval: '+ str(interval))
            plt.xlabel('Age')
            plt.ylabel('% Dead')
            cols = ac_df.columns
            for col in cols:
                plt.plot(ps.age_time, ac_df[col], label = col)

            plt.legend()
            
            plt.axis([25, 75, 0, 1])
            file_name = 'ac_death_'+ gene + '_'+ 'interval_constant_'+ str(interval) + '_combined_gender.png'
            if pf.check_valid_file(file_name) == False or overwrite_file == True:
                plt.savefig(file_name, dpi = 200)
            plt.show()
    return


def graph_everything_by_age(genes_every = ps.genes, genders_every = ps.genders, 
                            start_ages = ps.ages, intervals = [1],
                            overwrite_pngs = False):
    
    for i in intervals:
        
        CRC_results_by_age(genes = genes_every, genders = genders_every, 
                           interval = i, csy_start_ages = start_ages,
                           overwrite_file = overwrite_pngs)
        OS_results_by_age(genes = genes_every, genders = genders_every, 
                          csy_start_ages = start_ages,
                          interval = i,
                           overwrite_file = overwrite_pngs)
        CD_results_by_age(genes = genes_every, genders = genders_every,
                          csy_start_ages = start_ages,
                          interval = i,
                           overwrite_file = overwrite_pngs)
        AC_results_by_age(genes = genes_every, genders = genders_every,
                          csy_start_ages = start_ages,
                          interval = i,
                           overwrite_file = overwrite_pngs)

#graph_everything_by_age(overwrite_pngs = True)
'''
Generate outputs
All functions return a list of filenames that can be used to load files in ICER functions or main
 
'''
#generates .csv files for all strategies
#if export_csv == False, just returns a list of filenames to load for ICER
#(the purpose is to help cut down on computing time)
# overwrite_file = True is commented out
def generate_output_OWSA(gender, state_1, state_2, multiplier, intervals = ps.intervals):
    
    output_dict = {}
    for gene in ps.genes:
        for i in intervals:
            k = 0
            while k in range(0, len(ps.ages)):
                #if start age == 50, only test 5-year interval per current guidelines
# =============================================================================
#                 if ps.ages[k] == 50:
#                     i = 5
# =============================================================================
                run_spec = ps.run_type(i, gene, gender, age_spec = ps.ages[k])
                
# =============================================================================
#                 file_name = ('D_matrix_' + run_spec.gene + '_'+ str(run_spec.interval) + 
#                              '_' +str(run_spec.start_age) + '_both_genders.csv')
# =============================================================================
                key_name = ('D_matrix_' + run_spec.gene + '_'+ 
                               str(run_spec.interval) + '_' +
                               str(run_spec.start_age) + '_both_genders.csv')
                
                #checks if file w/outputs already exists, if not creates it, 
                #if the file already exists, returns filename w/o running whole simulation
#                if pf.check_valid_file(file_name) == False or overwrite_file == True:
#                    print('no valid file for run type, creating new one')
                D_matrix, t_matrix = run_markov_OWSA(run_spec, state_1, 
                                                       state_2, multiplier)
#                    D_matrix.to_csv(file_name, index = False)
                #print(file_name)
# =============================================================================
#                 if produce_csv == True:
#                     D_matrix, t_matrix = run_markov_simple(run_spec)
#                     D_matrix.to_csv(file_name, index = False)
# =============================================================================
#                print(D_matrix)
                output_dict.update({key_name: D_matrix})
# =============================================================================
#                 if i == intervals[0] and gene == ps.genes[0] and k == 0:
#                     filename_array = np.array([file_name])
#                     
#                 else:
#                     filename_array = np.append(filename_array, [file_name],
#                                                    axis = 0)
# =============================================================================
                if i == 0 and ps.ages[k] == 25:
                    k += 7
                else:  
                    k += 1
                    
#    filename_array = np.unique(filename_array)                    
    return output_dict


def generate_output(gender, intervals = ps.intervals):
    
    output_dict = {}
    for gene in ps.genes:
        for i in intervals:
            k = 0
            while k in range(0, len(ps.ages)):
                #if start age == 50, only test 5-year interval per current guidelines
# =============================================================================
#                 if ps.ages[k] == 50:
#                     i = 5
# =============================================================================
                run_spec = ps.run_type(i, gene, gender, age_spec = ps.ages[k])
                
                file_name = ('D_matrix_' + run_spec.gene + '_'+ str(run_spec.interval) + 
                             '_' +str(run_spec.start_age) + '_both_genders.csv')
                key_name = ('D_matrix_' + run_spec.gene + '_'+ 
                               str(run_spec.interval) + '_' +
                               str(run_spec.start_age) + '_both_genders.csv')
                
                #checks if file w/outputs already exists, if not creates it, 
                #if the file already exists, returns filename w/o running whole simulation
#                if pf.check_valid_file(file_name) == False or overwrite_file == True:
#                    print('no valid file for run type, creating new one')
                D_matrix, t_matrix = run_markov_simple(run_spec)
#                    D_matrix.to_csv(file_name, index = False)
                #print(file_name)
# =============================================================================
#                 if produce_csv == True:
#                     D_matrix, t_matrix = run_markov_simple(run_spec)
#                     D_matrix.to_csv(file_name, index = False)
# #                print(D_matrix)
# =============================================================================
                output_dict.update({key_name: D_matrix})
# =============================================================================
#                 if i == intervals[0] and gene == ps.genes[0] and k == 0:
#                     filename_array = np.array([file_name])
#                     
#                 else:
#                     filename_array = np.append(filename_array, [file_name],
#                                                    axis = 0)
# =============================================================================
                if i == 0 and ps.ages[k] == 25:
                    k += 7
                else:  
                    k += 1
                    
#    filename_array = np.unique(filename_array)                    
    return output_dict


def generate_output_PSA(gender, intervals = ps.intervals):
    
    output_dict = {}
    for gene in ps.genes:
        for i in intervals:
            k = 0
            while k in range(0, len(ps.ages)):
                #if start age == 50, only test 5-year interval per current guidelines
# =============================================================================
#                 if ps.ages[k] == 50:
#                     i = 5
# =============================================================================
                run_spec = ps.run_type(i, gene, gender, age_spec = ps.ages[k])
                
# =============================================================================
#                 file_name = ('D_matrix_' + run_spec.gene + '_'+ str(run_spec.interval) + 
#                              '_' +str(run_spec.start_age) + '_both_genders.csv')
# =============================================================================
                key_name = ('D_matrix_' + run_spec.gene + '_'+ 
                               str(run_spec.interval) + '_' +
                               str(run_spec.start_age) + '_both_genders.csv')
                
                #checks if file w/outputs already exists, if not creates it, 
                #if the file already exists, returns filename w/o running whole simulation
#                if pf.check_valid_file(file_name) == False or overwrite_file == True:
#                    print('no valid file for run type, creating new one')
                D_matrix, t_matrix = run_markov_PSA(run_spec)
#                    D_matrix.to_csv(file_name, index = False)
                #print(file_name)
# =============================================================================
#                 if produce_csv == True:
#                     D_matrix, t_matrix = run_markov_simple(run_spec)
#                     D_matrix.to_csv(file_name, index = False)
# =============================================================================
#                print(D_matrix)
                output_dict.update({key_name: D_matrix})
# =============================================================================
#                 if i == intervals[0] and gene == ps.genes[0] and k == 0:
#                     filename_array = np.array([file_name])
#                     
#                 else:
#                     filename_array = np.append(filename_array, [file_name],
#                                                    axis = 0)
# =============================================================================
                if i == 0 and ps.ages[k] == 25:
                    k += 7
                else:  
                    k += 1
                    
#    filename_array = np.unique(filename_array)                    
    return output_dict




#def generate_output_no_csv(gender,intervals = ps.intervals):
#
#    for gene in ps.genes:
#        for i in intervals:
#            k = 0
#            while k in range(0, len(ps.ages)):
#                #if start age == 50, only test 5-year interval per current guidelines
## =============================================================================
##                 if ps.ages[k] == 50:
##                     i = 5
## =============================================================================
#                run_spec = ps.run_type(i, gene, gender, age_spec = ps.ages[k])
#                D_matrix, t_matrix = run_markov_simple(run_spec)
#
#                if i == intervals[0] and gene == ps.genes[0] and k == 0:
#                    filename_array = np.array()
#                    
#                else:
#                    filename_array = np.append(filename_array, axis = 0)
#                if i == 0 and ps.ages[k] == 25:
#                    k += 7
#                else:  
#                    k += 1
#                    
#    filename_array = np.unique(filename_array)                    
#    return filename_array


#generate_output('male', overwrite_file = True)
#generate_output('female', overwrite_file = True)
#generates output for intervals but not starting ages
#can specify gender, start age, and gene, otherwise reverts to presets
def generate_output_lite_interval(genders = ps.genders, genes = ps.genes,
                                  age = ps.START_AGE, overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)        
        for gene in genes:
            for i in ps.intervals:
                run_spec = ps.run_type(i, gene, this_gender.gender, age_spec = age)
                #this_gender = g.gender_obj(run_spec.gender)
                #D_matrix, t_matrix = run_markov_simple(run_spec)
                
                file_append = run_spec.gene + '_' + run_spec.interval_str + '_' + gender +'_'+str(run_spec.start_age) + '_death.csv'
                file_name = 'D_matrix_' + file_append
                if pf.check_valid_file(file_name) == False or overwrite_file == True:
                    D_matrix, t_matrix = run_markov_simple(run_spec)
                    D_matrix.to_csv(file_name, index = False)
                    
                if gender == genders[0] and gene == genes[0] and i == 0:
                    filename_array = np.array([file_name])
                else:
                    filename_array = np.append(filename_array, [file_name], axis = 0)
    return filename_array

#defaults to interval = 1, can change to other intervals manually or get all intervals+ages from generate_output()
def generate_output_lite_start_age(genders = ps.genders, genes = ps.genes,
                                   interval = 1, overwrite_file = False):
    for gender in genders:
        this_gender = g.gender_obj(gender)        
        for gene in genes:
            i = 0
            for i in range(0, len(ps.ages)):
                run_spec = ps.run_type(interval, gene, this_gender.gender, age_spec = ps.ages[i])
                #this_gender = g.gender_obj(run_spec.gender)
                
                file_append = run_spec.gene + '_' + run_spec.interval_str + '_' + gender +'_'+str(run_spec.start_age) + '_death.csv'
                file_name = 'D_matrix_' + file_append
                
                if pf.check_valid_file(file_name) == False or overwrite_file == True:
                    D_matrix, t_matrix = run_markov_simple(run_spec)
                    D_matrix.to_csv(file_name, index = False)
                    
                if gender == genders[0] and gene == genes[0] and i == 0:
                    filename_array = np.array([file_name])
                else:
                    filename_array = np.append(filename_array, [file_name], axis = 0)
    return filename_array


