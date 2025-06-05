# lynch_icer
'''
Description: Functions to calculate ICERs and cost/QALY output
Author: Myles Ingram
Updated: 07.10.19
'''

import pandas as pd
import numpy as np
import lynch_presets as ps
import lynch_simulator as sim
#import PSA as sa
import probability_functions as pf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import data_manipulation as dm


def prob_ceiling(array):
    '''
    turns all values in array above 1 to 1
    '''
    for i in range(len(array)):
        if array[i] > 1:
            array[i] = 1
        else:
            continue
    return array


def unif_dist(df):
    '''
    iterates over the number of columns in the df and gives random numbers
    from a uniform distribution defined by +/_ the values of the columns of df
    '''
    for i in range(6, 17):
        col_val = df.values[:, i]
        if col_val[0] == 0:
            continue
        else:
            multiplier = np.random.uniform(.9, 1.10)
            new_vals = prob_ceiling(col_val * multiplier)
            df.iloc[:, i] = new_vals
    return df


def owsa_range(value):
    '''
    Gives upper and lower bounds of estimate using +/-
    '''
    upper = value + value*.2
    lower = value - value*.2
    return upper, lower


def owsa_range_list(array):
    upper_list = []
    lower_list = []
    for i in array:
        upper, lower = owsa_range(i)
        upper_list.append(upper)
        lower_list.append(lower)
    return upper_list, lower_list


def add_bounds_to_df(df, column):
# =============================================================================
#    Gives new df with upper and lower bound for all values in the given column
# =============================================================================
    new_df = df.copy()
    
    upper, lower = owsa_range_list(df[column])
    
    new_df.loc[:, "Upper_Bound"] = upper
    new_df.loc[:, "Lower_Bound"] = lower
    
    return new_df


def gamma_dist(old_list):
    k_list, th_list = [], []        
    for x in old_list:
        k, th = pf.gamma_dist(x, x*.3)
        k_list.append(k)
        th_list.append(th)
    new_list = [np.random.gamma(k, th) for k, th in zip(k_list, th_list)]
    return new_list


def beta_dist(old_list):
    a_list, b_list = [], []
    for x in old_list:
        a, b = pf.beta_dist(x, np.sqrt(x*(1-x)))
        a_list.append(a)
        b_list.append(b)
    new_list = [np.random.beta(a, b) for a, b in zip(a_list, b_list)]
    return new_list


def costs_dist():
    new_costs = ps.costs.copy()
    new_costs.loc[:, "Cost"] = gamma_dist(ps.costs.loc[:, "Cost"])
    
    return new_costs


def util_dist():
    
    df_util = unif_dist(ps.utils_m.copy())
    
    return df_util




    
def calculate_utility_table(D_matrix, utility_table):
#    if gender == 'male':
#        utility_table = ps.utils_m.copy()
#    else:

    utility_results = utility_table.copy()
    healthy_states = ['mutation', 'current', 'new', 'nono']
    bx_states = ['init adenoma']
    comp_tracker = np.zeros(len(D_matrix))
    csy_num = np.zeros(len(D_matrix))
    #D_matrix['csy comps'] = 0
    i = 0
    for i in range(0, len(D_matrix)):
        k = 0
        #loop over columns in i-th row to get utilities * prop in state k
        for k in range(0, 18):
            utility_results.iloc[i, k] = utility_table.iloc[i, k] * D_matrix.iloc[i, k]
            if utility_results.iloc[i, k] < 0:
                print('error, utility results < 0', i, k)
            
        cycle_comp_tracker = 0.
        
        if D_matrix.loc[i, 'csy tracker'] == True:
            #prop_comp = D_matrix.loc[i, run_spec.guidelines] * ps.csy_comp
            #if csy, this year, apply disutil to whichever state isn't empty
            #disutility for csy given adenoma already applied
            for col in healthy_states:
                if D_matrix.loc[i, col] != 0:
                    cycle_csy_disutil = ps.csy_disutil * D_matrix.loc[i, col]
                   # D_matrix.loc[i, 'csy comps'] += ps.p_csy_comp_no_bx * D_matrix.loc[i, col]
                   
                    #multiply complication disutility by proportion with complications this cycle
                    cycle_csy_comp_disutil = ps.compl_disutil * ps.p_csy_comp_no_bx * D_matrix.loc[i, col]
                    cycle_comp_tracker += ps.p_csy_comp_no_bx * D_matrix.loc[i, col]
                    csy_num[i] += D_matrix.loc[i, col]
                    #print('first util =', utility_results.loc[i, col])
                    #subtract disutil of csy and csy comps from this utility
                    utility_results.loc[i, col] = utility_results.loc[i, col] - cycle_csy_disutil - cycle_csy_comp_disutil
                
                    #print('minus csy =', utility_results.loc[i, col])
                if utility_results.loc[i, col] < 0:
                    print('error, utility results <0')
            
        
        #apply adenoma disutility comps regardless of csy tracker. assume no bx
        adenoma_comps = ps.p_csy_comp_no_bx * D_matrix.loc[i, 'adenoma']
        csy_num[i] += D_matrix.loc[i, 'adenoma']
        csy_num[i] += D_matrix.loc[i, 'init adenoma']
        cycle_comp_tracker += adenoma_comps
        utility_results.loc[i, 'adenoma'] -= adenoma_comps * ps.compl_disutil
        if utility_results.loc[i, 'adenoma'] < 0:
            print('error, utility results <0')
        #apply disutility to larger prop. w/comps who had bx
        for b in bx_states:
            if D_matrix.loc[i, b] != 0:
                #get disutility w higher comp prob
                cycle_csy_comps = ps.p_csy_comp_bx * D_matrix.loc[i, b]
                cycle_comp_tracker += cycle_csy_comps
                cycle_disutil = cycle_csy_comps * ps.compl_disutil
                utility_results.loc[i, b] -= cycle_disutil
                
            if utility_results.loc[i, b] < 0:
                print('error, utility results <0')
        comp_tracker[i] = cycle_comp_tracker
                 
    
    
    return(utility_results)

def calculate_cost_table(D_matrix, costs):
    
    cost_table = D_matrix.drop(columns = ['csy tracker'])
    healthy_states = ['mutation', 'current', 'new', 'nono']
    
    cancer_states = ['init dx stage I', 'init dx stage II', 'init dx stage III',
                     'init dx stage IV', 'dx stage I', 'dx stage II', 'dx stage III',
                     'dx stage IV', 'stage I death',
                    'stage II death', 'stage III death', 
                    'stage IV death']
    death_states = ['all cause dx', 'cancer death',
                    'all cause', 'csy death']
    
    i = 0
    
    for i in range(0, len(cost_table)):
        #if csy this cycle, apply costs
        if D_matrix.loc[i, 'csy tracker'] == True:
            for col in healthy_states:
                if D_matrix.loc[i, col]!= 0:
                    cost_table.loc[i, col] = D_matrix.loc[i, col] * costs.loc['csy_no_bx', 'Cost']
                    
                    csy_comps = D_matrix.loc[i, col] * ps.p_csy_comp_no_bx
                    cost_table.loc[i, col] += (csy_comps * costs.loc['csy comp', 'Cost'])
        else:
            for col in healthy_states:
                cost_table.loc[i, col] = 0
        cost_table.loc[i, 'adenoma'] = D_matrix.loc[i, 'adenoma'] * costs.loc['csy_no_bx', 'Cost']
        #get proportion of adenoma pts with complication, assume no bx
        csy_comp_adenoma = D_matrix.loc[i, 'adenoma'] * ps.p_csy_comp_no_bx
        cost_table.loc[i, 'adenoma'] += csy_comp_adenoma * costs.loc['csy comp', 'Cost']
        
        #apply cost for csy with bx/polypectomy
        cost_table.loc[i, 'init adenoma'] = D_matrix.loc[i, 'init adenoma'] * costs.loc['csy_bx_polyp', 'Cost']
        #add cost for complications
        cost_table.loc[i, 'init adenoma'] += D_matrix.loc[i, 'init adenoma'] * ps.p_csy_comp_bx *costs.loc['csy comp', 'Cost']
        
        #print('colonoscopy cost', cost_table.loc[i, 'mutation']+cost_table.loc[i, 'new']+cost_table.loc[i, 'current']+cost_table.loc[i, 'adenoma'])
        for col in cancer_states:
            cost_table.loc[i, col] = D_matrix.loc[i, col] * costs.loc[col, 'Cost']
        for col in death_states:
            cost_table.loc[i, col] = D_matrix.loc[i, col] * 0
        
   
    return cost_table


def calculate_LE_table(D_matrix):
    
    LE_results = D_matrix.drop(columns = ['csy tracker'])
    alive_states = ['mutation', 'current', 'new', 'nono', 'init adenoma', 'adenoma', 'init dx stage I',
                      'init dx stage II', 'init dx stage III', 'init dx stage IV',
                      'dx stage I', 'dx stage II', 'dx stage III', 'dx stage IV']
    death_states = ['all cause', 'all cause dx', 'cancer death', 'csy death', 'stage 1 death',
                    'stage 2 death', 'stage 3 death', 'stage 4 death']
    i = 0
    for i in range(0, len(D_matrix)):
        for col in alive_states:
            LE_results.loc[i, col] = D_matrix.loc[i, col] * 1
        for col in death_states:
            LE_results.loc[i, col] = 0
                    
    return(LE_results)


def create_ce_table(D_matrix, costs, utility):
# =============================================================================
#     Creates a dataframe with stage-by-stage QALYs (disc & undisc.), costs (disc & undisc.), 
#       and LE. 
# =============================================================================
    D_matrix_temp = D_matrix.drop(columns = ['cancer incidence', 'overall survival','age'])
    utility_table = calculate_utility_table(D_matrix_temp, utility)
    #D_matrix['comp tracker'] = utility_table['comp tracker']
    cost_table = calculate_cost_table(D_matrix_temp, costs)
    LE_table = calculate_LE_table(D_matrix_temp)
    
#   Presets for the loop
    Range = D_matrix.index

    for t in Range:

        if t == Range[0]:
            QALY_results = np.array([sum(utility_table.loc[t, :])])
            QALY_results_d = np.array([pf.discount(QALY_results[t], t, input_type = 'QALY')])
            
            LE_results = np.array([sum(LE_table.loc[t, :])])
            
            cost_results = np.array([sum(cost_table.loc[t, :])])
            cost_results_d = np.array([pf.discount(LE_results[t], t)])
        else:
            QALY_results = np.append(QALY_results, [sum(utility_table.loc[t, :])],
                                                    axis = 0)
            QALY_results_d = np.append(QALY_results_d, [pf.discount(QALY_results[t], t, input_type = 'QALY')],
                                       axis = 0)
            
            LE_results = np.append(LE_results, [sum(LE_table.loc[t, :])],
                                                axis = 0)
            
            cost_results = np.append(cost_results, [sum(cost_table.loc[t, :])],
                                                    axis = 0)
            cost_results_d = np.append(cost_results_d, [pf.discount(cost_results[t], t)],
                                                        axis = 0)

    ce_table = pd.DataFrame({'QALYs': QALY_results, 'disc QALYs': QALY_results_d,
                             'LE': LE_results, 'Costs': cost_results, 
                             'disc Costs':cost_results_d})
    return ce_table

#ce_table = create_ce_table(D_matrix_test, 'male')

def calculate_ce_results(ce_table):
    #Takes stage-by-stage results from create_ce_table and sums for totals
    
    #QALYs = sum(ce_table.loc[:, 'QALYs'])
    QALYs = sum(ce_table.loc[:, 'disc QALYs'])
    LE = sum(ce_table.loc[:, 'LE'])
    costs = sum(ce_table.loc[:, 'disc Costs'])
    
    return QALYs, LE, costs


#runs through every interval of given genders, genes, start age
#first two inputs must be arrays even if only one element!!
def generate_interval_ce_tables(genders_sim = ps.genders, genes_sim = ps.genes, 
                                age_sim = ps.START_AGE):
    filenames = sim.generate_output_lite_interval(genders = genders_sim, genes= genes_sim,
                                                  age = age_sim)
    run_array = filenames
    QALY_array = np.zeros(len(filenames))
    LE_array = np.zeros(len(filenames))
    cost_array = np.zeros(len(filenames))
    
    
    i = 0
    for f in filenames:
        D_matrix = pd.read_csv(f)
        print('running filename', f)
        if 'female' in f:
            gender = 'female'
        else:
            gender = 'male'
        if f == filenames[0]:
            gene = np.array([f[9:9+4]])
        else:
            gene = np.append(gene, [f[9:9+4]])
        
        QALY_array[i], LE_array[i], cost_array[i] = calculate_ce_results(create_ce_table(D_matrix, gender))
        print('QALYs', QALY_array[i])
        print('LYs', LE_array[i])
        i += 1
    all_result_df = pd.DataFrame({'gene': gene, 'run type': run_array, 'QALYs':QALY_array,
                                  'LE': LE_array, 'cost': cost_array})
    return all_result_df

#lite_df = generate_interval_ce_tables(genders_sim = ['male'])
#lite_df.to_csv('lite_output_test.csv')
    
def get_num_csy(D_matrix):
    num_csy = 0
    i = 0
    csy_states = ['new', 'current']
    for i in range(0, len(D_matrix)):
        if D_matrix.loc[i, 'csy tracker'] == True:
            for col in csy_states:
                num_csy += D_matrix.loc[i, col]
        num_csy += D_matrix.loc[i, 'init adenoma']
        num_csy += D_matrix.loc[i, 'adenoma']
    return num_csy

def get_num_comps(D_matrix):
    num_csy = 0
    i = 0
    csy_states = ['new', 'current']
    for i in range(0, len(D_matrix)):
        if D_matrix.loc[i, 'csy tracker'] == True:
            for col in csy_states:
                num_csy += (D_matrix.loc[i, col] * ps.p_csy_comp_no_bx)
        num_csy += (D_matrix.loc[i, 'init adenoma'] * ps.p_csy_comp_bx)
        num_csy += (D_matrix.loc[i, 'adenoma'] * ps.p_csy_comp_no_bx)
    return num_csy


#runs through every scenario for a given gender 
# =============================================================================
# def generate_all_ce_tables(utility, cost, overwrite = True):
#     filenames = sim.generate_output("both", overwrite_file = overwrite)
#     
#     df = pd.DataFrame(columns = ['gene', 'Strategy', 'csy interval',
#                                  'csy start age', 'run type', 'QALYs',
#                                  'Life-years', 'CRC incidence', 'CRC death', 'cost',
#                                  'colonoscopies', 'csy comps'], index = range(0, len(filenames)))
#     
#     i = 0
#     for f in filenames:
#         D_matrix = pd.read_csv(f)
#         #print('running filename', f)
#         if f == filenames[0]:
#             gene = np.array([f[9:9+4]])
#         else:
#             gene = np.append(gene, [f[9:9+4]])
#         df.loc[i, 'gene'] = f[9:9+4]
#         #QALY_array[i], LE_array[i], cost_array[i] = calculate_ce_results(create_ce_table(D_matrix, gender))
#         df.loc[i, 'QALYs'], df.loc[i, 'Life-years'], df.loc[i, 'cost'] = calculate_ce_results(create_ce_table(D_matrix, cost, utility))
#         #cancer_incidence[i] = D_matrix.loc[len(D_matrix)-1, 'cancer incidence']
#         df.loc[i, 'CRC incidence'] = D_matrix.loc[len(D_matrix)-1, 'cancer incidence']
#         df.loc[i, 'CRC death'] = D_matrix.loc[len(D_matrix)-1, 'cancer death']
#         df.loc[i, 'colonoscopies'] = get_num_csy(D_matrix)
#         df.loc[i, 'csy comps'] = get_num_comps(D_matrix)
#         df.loc[i, 'csy start age'] =f[16:18]
#         df.loc[i, 'csy interval'] = f[14]
#         df.loc[i, 'Strategy'] = 'Q'+str(f[14])+'Y, Start age: '+ str(f[16:18])
#         
#         i += 1
#     df['run type'] = filenames
#     output_file = 'all_ce_results_combined_gender.csv'
#     
#     df.to_csv(output_file, index = False)
#     
#     return df
# 
# =============================================================================
def generate_all_ce_tables(utility, cost):
    # Run simluation
    output_dict = sim.generate_output("both") 
    
    df = pd.DataFrame(columns = ['gene', 'Strategy', 'csy interval',
                                 'csy start age', 'run type', 'QALYs',
                                 'Life-years', 'CRC incidence', 'CRC death', 'cost',
                                 'colonoscopies', 'csy comps'], index = range(0, len(output_dict)))
    
    i = 0
#    for f in filenames:
    for key in output_dict.keys():
        D_matrix = output_dict[key]
#        D_matrix = pd.read_csv(f)
        #print('running filename', f)
#        if f == filenames[0]:
        if i == 0:
            gene = np.array([key[9:9+4]])
        else:
            gene = np.append(gene, [key[9:9+4]])
        df.loc[i, 'gene'] = key[9:9+4]
        #QALY_array[i], LE_array[i], cost_array[i] = calculate_ce_results(create_ce_table(D_matrix, gender))
        df.loc[i, 'QALYs'], df.loc[i, 'Life-years'], df.loc[i, 'cost'] = calculate_ce_results(create_ce_table(D_matrix, cost, utility))
        #cancer_incidence[i] = D_matrix.loc[len(D_matrix)-1, 'cancer incidence']
        df.loc[i, 'CRC incidence'] = D_matrix.loc[len(D_matrix)-1, 'cancer incidence']
        df.loc[i, 'CRC death'] = D_matrix.loc[len(D_matrix)-1, 'cancer death']
        df.loc[i, 'colonoscopies'] = get_num_csy(D_matrix)
        df.loc[i, 'csy comps'] = get_num_comps(D_matrix)
        df.loc[i, 'csy start age'] = key[16:18]
        df.loc[i, 'csy interval'] = key[14]
        df.loc[i, 'Strategy'] = 'Q'+str(key[14])+'Y, Start age: '+ str(key[16:18])
        
        i += 1
    df['run type'] = output_dict.keys()
    output_file = 'all_ce_results_combined_gender_v2.csv'
    
    df.to_csv(output_file, index = False)
    
    return df


def generate_all_ce_tables_PSA(utility, cost):
    output_dict = sim.generate_output_PSA("both") 
    
    df = pd.DataFrame(columns = ['gene', 'Strategy', 'csy interval',
                                 'csy start age', 'run type', 'QALYs',
                                 'Life-years', 'CRC incidence', 'CRC death', 'cost',
                                 'colonoscopies', 'csy comps'], index = range(0, len(output_dict)))
    
    i = 0
#    for f in filenames:
    for key in output_dict.keys():
        D_matrix = output_dict[key]
#        D_matrix = pd.read_csv(f)
        #print('running filename', f)
#        if f == filenames[0]:
        if i == 0:
            gene = np.array([key[9:9+4]])
        else:
            gene = np.append(gene, [key[9:9+4]])
        df.loc[i, 'gene'] = key[9:9+4]
        #QALY_array[i], LE_array[i], cost_array[i] = calculate_ce_results(create_ce_table(D_matrix, gender))
        df.loc[i, 'QALYs'], df.loc[i, 'Life-years'], df.loc[i, 'cost'] = calculate_ce_results(create_ce_table(D_matrix, cost, utility))
        #cancer_incidence[i] = D_matrix.loc[len(D_matrix)-1, 'cancer incidence']
        df.loc[i, 'CRC incidence'] = D_matrix.loc[len(D_matrix)-1, 'cancer incidence']
        df.loc[i, 'CRC death'] = D_matrix.loc[len(D_matrix)-1, 'cancer death']
        df.loc[i, 'colonoscopies'] = get_num_csy(D_matrix)
        df.loc[i, 'csy comps'] = get_num_comps(D_matrix)
        df.loc[i, 'csy start age'] = key[16:18]
        df.loc[i, 'csy interval'] = key[14]
        df.loc[i, 'Strategy'] = 'Q'+str(key[14])+'Y, Start age: '+ str(key[16:18])
        
        i += 1
    df['run type'] = output_dict.keys()
#    output_file = 'all_ce_results_combined_gender.csv'
    
#    df.to_csv(output_file, index = False)
    
    return df


def generate_all_ce_tables_prob(utility, cost, state_1, state_2, multiplier):
    output_dict = sim.generate_output_OWSA("both", state_1, state_2, multiplier) 
    
    df = pd.DataFrame(columns = ['gene', 'Strategy', 'csy interval',
                                 'csy start age', 'run type', 'QALYs',
                                 'Life-years', 'CRC incidence', 'CRC death', 'cost',
                                 'colonoscopies', 'csy comps'], index = range(0, len(output_dict)))
    
    i = 0
#    for f in filenames:
    for key in output_dict.keys():
        D_matrix = output_dict[key]
#        D_matrix = pd.read_csv(f)
        #print('running filename', f)
#        if f == filenames[0]:
        if i == 0:
            gene = np.array([key[9:9+4]])
        else:
            gene = np.append(gene, [key[9:9+4]])
        df.loc[i, 'gene'] = key[9:9+4]
        #QALY_array[i], LE_array[i], cost_array[i] = calculate_ce_results(create_ce_table(D_matrix, gender))
        df.loc[i, 'QALYs'], df.loc[i, 'Life-years'], df.loc[i, 'cost'] = calculate_ce_results(create_ce_table(D_matrix, cost, utility))
        #cancer_incidence[i] = D_matrix.loc[len(D_matrix)-1, 'cancer incidence']
        df.loc[i, 'CRC incidence'] = D_matrix.loc[len(D_matrix)-1, 'cancer incidence']
        df.loc[i, 'CRC death'] = D_matrix.loc[len(D_matrix)-1, 'cancer death']
        df.loc[i, 'colonoscopies'] = get_num_csy(D_matrix)
        df.loc[i, 'csy comps'] = get_num_comps(D_matrix)
        df.loc[i, 'csy start age'] = key[16:18]
        df.loc[i, 'csy interval'] = key[14]
        df.loc[i, 'Strategy'] = 'Q'+str(key[14])+'Y, Start age: '+ str(key[16:18])
        
        i += 1
    df['run type'] = output_dict.keys()
    output_file = 'all_ce_results_combined_gender.csv'
    
    df.to_csv(output_file, index = False)
    
    return df



def get_icers(result_df):
    #INPUT: dataframe produced by generate_ce functions
    #OUTPUT: ce_table with dominated strategies eliminated and icers added
    # Order input table by ascending cost
    ce_icers = result_df.sort_values(by=['cost'])
    #placeholder to keep results from other strategies
    
    
#    print(ce_icers)
#    print('calculating ICERs')
    ce_icers = ce_icers.reset_index(drop = True)
    
    
    num_rows = len(ce_icers)
    row = 0
    # Eliminate strongly dominated strategies (lower qalys; higher cost)
    while row < num_rows-1:
        if(ce_icers['QALYs'][row+1] < ce_icers['QALYs'][row]):
            ce_icers = ce_icers.drop([ce_icers.index[row+1]])
            ce_icers = ce_icers.reset_index(drop = True)
            num_rows = len(ce_icers)
            row = 0
        else:
            row += 1
          
    # Initiate icers column
    ce_icers.loc[:, 'icers'] = 0.
    
    
    # Calculate remaining icers and eliminate weakly dominated strategies
    if len(ce_icers) > 1:
        num_rows = len(ce_icers)
        row = 1
        while row < num_rows:
            # Calculate icers
            ce_icers.loc[ce_icers.index[row], 'icers'] = (
                    (ce_icers['cost'][row]-ce_icers['cost'][row-1]) / 
                    (ce_icers['QALYs'][row]-ce_icers['QALYs'][row-1]))
            ce_icers.loc[ce_icers.index[row], 'icers'] = np.round(ce_icers.loc[ce_icers.index[row], 'icers'], 
                                                                    decimals = 2)
#            print(ce_icers)
            # If lower qaly and higher icer, eliminate strategy
            if(ce_icers.loc[ce_icers.index[row], 'icers'] < 
               ce_icers.loc[ce_icers.index[row-1], 'icers']):
                ce_icers = ce_icers.drop([ce_icers.index.values[row-1]])
                ce_icers = ce_icers.reset_index(drop = True)
                num_rows = len(ce_icers)
                row = row - 1
            else:
                row += 1
            
            
    return ce_icers

def get_icers_with_dominated(result_df):
    #INPUT: dataframe produced by generate_ce functions
    #OUTPUT: ce_table with dominated strategies eliminated and icers added
    # Order input table by ascending cost
    ce_icers = result_df.sort_values(by=['cost'])
    #placeholder to keep results from other strategies
    
#    print(ce_icers)
#    print('calculating ICERs')
    ce_icers = ce_icers.reset_index(drop = True)
    num_rows = len(ce_icers)
    row = 0
    ce_icers_temp = ce_icers.copy()
    # Eliminate strongly dominated strategies (lower qalys; higher cost)
    while row < num_rows-1:
        if(ce_icers['QALYs'][row+1] < ce_icers['QALYs'][row]):
            ce_icers = ce_icers.drop([ce_icers.index[row+1]])
            ce_icers = ce_icers.reset_index(drop = True)
            num_rows = len(ce_icers)
            row = 0
        else:
            row += 1
          
    # Initiate icers column
    ce_icers.loc[:, 'icers'] = 0
    # Calculate remaining icers and eliminate weakly dominated strategies
    if len(ce_icers) > 1:
        num_rows = len(ce_icers)
        row = 1
        while row < num_rows:
            # Calculate icers
            ce_icers.loc[ce_icers.index[row], 'icers'] = (
                    (ce_icers['cost'][row]-ce_icers['cost'][row-1]) / 
                    (ce_icers['QALYs'][row]-ce_icers['QALYs'][row-1]))
#            print(ce_icers)
            # If lower qaly and higher icer, eliminate strategy
            if(ce_icers.loc[ce_icers.index[row], 'icers'] < 
               ce_icers.loc[ce_icers.index[row-1], 'icers']):
                ce_icers = ce_icers.drop([ce_icers.index.values[row-1]])
                ce_icers = ce_icers.reset_index(drop = True)
                num_rows = len(ce_icers)
                row = row - 1
            else:
                row += 1
    ce_icers_temp = ce_icers.set_index('run type', drop = True)
    ce_icers_full = result_df.copy()
    ce_icers_full['cost per QALY'] = 0.
    ce_icers_full['icers'] = 0.
    
    #to keep track of run type 
    ce_icers_full['csy interval'] = 0.
    ce_icers_full['csy start age'] = 0.
    for i in range(0, len(ce_icers_full)):
        if ce_icers_full.loc[i, 'run type'] in ce_icers_temp.index:
            ce_icers_full.loc[i, 'icers'] = ce_icers_temp.loc[ce_icers_full.loc[i, 'run type'], 'icers']
            ce_icers_full.loc[i, 'icers'] = np.round(ce_icers_full.loc[i, 'icers'], decimals = 2)
        else:
            ce_icers_full.loc[i, 'icers'] = 'dominated'
        ce_icers_full.loc[i, 'cost per QALY'] = ce_icers_full.loc[i, 'cost']/ce_icers_full.loc[i, 'QALYs']
        run_type = ce_icers_full.loc[i, 'run type']
        ce_icers_full.loc[i, 'csy interval'] = run_type[14]
        ce_icers_full.loc[i, 'csy start age'] = run_type[16:18]
    return ce_icers_full


def get_icer_gene(result_df):
    """Get ICER for each gene

    Args:
        result_df (DataFrame): ICERs for each gene
    """
    icers_list = []
    for gene in ps.genes:
#        print('generating efficiency frontier for', gene)
        this_df = result_df[result_df['gene'] == gene]
        this_df = this_df.reset_index(drop = True)
        #get icers
        icers = get_icers(this_df)
        icers_list.append(icers)
    return icers_list


def generate_ICER_with_plots_fancy(result_df):
    #INPUT: All cost and effectiveness data produced by generate_ce function
    #OUTPUT: efficiency frontier graphs and ICER .csv files for each gene
    #(221), (222), (223), and (224),
    
    for gene in ps.genes:
        print('generating efficiency frontier for', gene)
        this_df = result_df[result_df['gene'] == gene]
        this_df = this_df.reset_index(drop = True)
        #get icers
        icers = get_icers(this_df)
        icers_full = get_icers_with_dominated(this_df)
        fig, ax = plt.subplots()
        
        #ax1 = ax.twinx
        x = this_df['cost'].values.tolist()
        y = this_df['QALYs'].values.tolist()
        labels = this_df['run type'].values.tolist()
        colors = np.empty(len(x), dtype = str)
        ages = np.empty(len(x), dtype = int)
        #extract run type info from file name
        i = 0
        for i in range(0, len(labels)):
            new_label = labels[i]
            old_label = labels[i]
            new_label = 'Q' + str(old_label[14]) + 'Y' + ' age ' + str(old_label[16:18])
            if i == 0:
                gender_label = old_label[19]
            labels[i] = new_label
            interval = new_label[0:3]
            if i == 0:
                intervals = np.array([interval])
                
                interval_legend = list()
                #legend_counter = 0
            else:
                intervals = np.append(intervals, [interval], axis = 0)
            
            if 'Q1Y' in labels[i]:
                intervals[i] = 'Q1Y'
                colors[i] = 'b'
                #interval_1 = mpatches.Patch(color=colors[i], label=intervals[i])
                
                #interval_legend[i] = mpatches.Patch(color=colors[i], label=intervals[i])
                #legend_counter += 1
            elif 'Q2Y' in labels[i]:
                intervals[i] = 'Q2Y'
                colors[i] = 'r'
                #interval_2 = mpatches.Patch(color=colors[i], label=intervals[i])
                
            elif 'Q3Y' in labels[i]:
                intervals[i] = 'Q3Y'
                colors[i] = 'g'
                #interval_3 = mpatches.Patch(color=colors[i], label=intervals[i])
            
            elif 'Q4Y' in labels[i]:
                intervals[i] = 'Q4Y'
                colors[i] = 'y'
                #interval_4 = mpatches.Patch(color=colors[i], label=intervals[i])
            
            elif 'Q5Y' in labels[i]:
                intervals[i] = 'Q5Y'
                colors[i] = 'm'
                #interval_5 = mpatches.Patch(color=colors[i], label=intervals[i])
            #interval_legend.append(mpatches.Patch(color=colors[i], label=intervals[i]))
            if '25' in labels[i]:
                ages[i] = 25
                
            elif '30' in labels[i]:
                ages[i] = 30
            
            elif '35' in labels[i]:
                ages[i] = 35
            
            elif '40' in labels[i]:
                ages[i] = 40
            
            elif '45' in labels[i]:
                ages[i] = 45
               
            elif '50' in labels[i]:
                ages[i] = 50
        plt.scatter(x, y, c = colors)
        for j, txt in enumerate(ages):
            plt.annotate(txt, (x[j], y[j]))
        temp_colors = np.unique(colors)
        
        temp_intervals = np.unique(intervals)
        temp_intervals = ['Q1Y', 'Q2Y', 'Q3Y', 'Q4Y', 'Q5Y']
        temp_colors = ['b', 'r', 'g', 'y', 'm']
        k = 0
        for k in range(0, len(temp_colors)):
            interval_legend.append(mpatches.Patch(color = temp_colors[k], label = temp_intervals[k]))
        plt.legend(handles = interval_legend)
        
        
        plot_title = 'QALYs and Costs: ' + gene
        plt.ylabel('QALYs')
        plt.xlabel('Cost')
        plt.title(plot_title)
        plot_name = 'eff_frontier_' + gene +'_combined_gender.png'
        
        plt.savefig(plot_name, dpi = 300)
        plt.show()
        
        #icers = get_icers(this_df)
        #icers_full = get_icers_with_dominated(this_df)
        icers = icers.rename(index=str, columns={"icers": "ICERs"})
        
        filename = 'ICERS_'+gene+'_combined_gender.csv'
        filename_full = 'ICERS_with_dominated_combined_gender_'+gene+'.csv'
        #only creates new file if no file w name already exists
        icers.to_csv(filename, index = False)
        
        icers_full.to_csv(filename_full, index = False)
        print('-'*30)
        print(gene)
        print(icers)
    plt.show()
    
def graph_eff_frontiers(full_df):
    
    for gene in ps.genes:
        temp_df = full_df[full_df['gene'] == gene]
        temp_df = temp_df.reset_index(drop = True)
        
        this_df = get_icers(temp_df)
        fig, ax = plt.subplots()
        i = 0
        for i in range(0, len(this_df)):
            this_run_type = ('  Start age: ' + str(this_df.loc[i, 'csy start age']) + ', Interval: ' +
                                 str(this_df.loc[i, 'csy interval']))
            if i == 0:
                
                labels = np.array([this_run_type])
            else:
                labels = np.append(labels, [this_run_type])
        
        plt.ylabel('QALYs')
        plt.xlabel('Cost')
        num_rows = len(this_df)
        if num_rows > 1:
            for i in range(0, num_rows - 1):
                line_x = [this_df.loc[i, 'cost'], this_df.loc[i+1, 'cost']]
                line_y = [this_df.loc[i, 'QALYs'], this_df.loc[i+1, 'QALYs']]
                line_text = '${:,.2f}'.format(this_df.loc[i+1, 'icers'])
                line_label_x = ((line_x[1] + line_x[0])/2) + 100
                line_label_y = ((line_y[1] + line_y[0])/2) - .0008
                plt.plot(line_x, line_y, label = line_text, color = 'b')
                plt.text(line_label_x, line_label_y, line_text)
            y_vals = this_df['QALYs']
            x_vals = this_df['cost']
        
            ax.scatter(x_vals, y_vals, label = labels)
            for i, txt in enumerate(labels):
                ax.annotate(txt, (x_vals[i] + 1, y_vals[i]))
            
            #plt.legend()
        else:
            x = this_df.loc[0, 'cost']
            x = int(x)
            y = this_df.loc[0, 'QALYs']
            print(this_df['cost'])
            print(x)
            x_ticks = np.arange(x-200, x+201, 100, dtype = int)
            #x_ticks = [str(int(x - 100)), str(int(x)), str(int(x + 100))]
            #ax.set_xticklabels(x_ticks)
            ax.set_xbound(lower = min(x_ticks), upper = max(x_ticks))
            ax.scatter(x, y, label = this_run_type)
            ax.annotate(this_run_type, (x, y))
            
        plt.title('Efficiency Frontier: ' + this_df.loc[0, 'gene'])
        
        plt.savefig(gene+'_combined_genders_frontier_clean.png', bbox_inches = 'tight', dpi = 200)
        plt.show()


def run_CEA():
    utility = ps.utils_m.copy()
#    print(type(utility))
    cost = ps.costs.copy()
    df = generate_all_ce_tables(utility, cost)
    results = get_icer_gene(df)
#    generate_ICER_with_plots_fancy(df)
#    pd.DataFrame(results).to_csv("icer_results_table.csv")
    return results

print(run_CEA())

def run_CEA_PSA():
    utility = util_dist()
#    print(type(utility))
    cost = costs_dist()
    df = generate_all_ce_tables_PSA(utility, cost)
    results = get_icer_gene(df)
    generate_ICER_with_plots_fancy(df)
#    pd.DataFrame(results).to_csv("icer_results_table.csv")
    return results


def generate_icers_all_genes(utility, cost):
    df = generate_all_ce_tables(utility, cost)
    icers_list = []
    optimal_strat = []
    optimal_icer = []
    for gene in ps.genes:
        if gene == "MLH1":
            strat = ["Q1Y, Start age: 25"]
        elif gene == "MSH2":
            strat = ["Q2Y, Start age: 25"]
        elif gene == "MSH6":
            strat = ["Q3Y, Start age: 40"]
        else:
            strat = ["Q3Y, Start age: 40"]
#       print('generating efficiency frontier for', gene)
        this_df = df[df['gene'] == gene]
        this_df = this_df.reset_index(drop = True)
        #get icers
        icers = get_icers(this_df)
        opt_icers = icers[icers["icers"] <= 100000]
        optimal_strat.append(opt_icers.iloc[-1, :]["Strategy"])
        optimal_icer.append(opt_icers.iloc[-1, :]["icers"])
        icer_arr = dm.selection(icers, strat, "Strategy")
        if len(icer_arr) == 0:
            icer_val = "N/A"
        else:
            icer_val = icer_arr["icers"].values[0]
        icers_list.append(icer_val)
    return icers_list, optimal_strat, optimal_icer



def generate_icers_all_genes_prob(utility, cost, state_1, state_2, multiplier):
    df = generate_all_ce_tables_prob(utility, cost, state_1, state_2, multiplier)
    icers_list = []
    for gene in ps.genes:
        if gene == "MLH1":
            strat = ["Q1Y, Start age: 25"]
        elif gene == "MSH2":
            strat = ["Q2Y, Start age: 25"]
        elif gene == "MSH6":
            strat = ["Q3Y, Start age: 40"]
        else:
            strat = ["Q3Y, Start age: 40"]
#       print('generating efficiency frontier for', gene)
        this_df = df[df['gene'] == gene]
        this_df = this_df.reset_index(drop = True)
        #get icers
        icers = get_icers(this_df)
        icer_arr = dm.selection(icers, strat, "Strategy")
        if len(icer_arr) == 0:
            icer_val = "N/A"
        else:
            icer_val = icer_arr["icers"].values[0]
        icers_list.append(icer_val)
    return icers_list
    


def run_owsa_CEA_cost():
    utility = ps.utils_m.copy()
#    print(type(utility))
    cost = add_bounds_to_df(ps.costs, "Cost")
#   icer lists (make all code below more efficient)
    # dumb way of doing this want to get this project over with
    MLH1_upper_icer = []
    MLH1_lower_icer = []
    MSH2_upper_icer = []
    MSH2_lower_icer = []
    MSH6_upper_icer = []
    MSH6_lower_icer = []
    PMS2_upper_icer = []
    PMS2_lower_icer = []
    MLH1_opt_upper_icer = []
    MLH1_opt_lower_icer = []
    MSH2_opt_upper_icer = []
    MSH2_opt_lower_icer = []
    MSH6_opt_upper_icer = []
    MSH6_opt_lower_icer = []
    PMS2_opt_upper_icer = []
    PMS2_opt_lower_icer = []
    MLH1_opt_upper_strat = []
    MLH1_opt_lower_strat = []
    MSH2_opt_upper_strat = []
    MSH2_opt_lower_strat = []
    MSH6_opt_upper_strat = []
    MSH6_opt_lower_strat = []
    PMS2_opt_upper_strat = []
    PMS2_opt_lower_strat = []
    for row in cost.itertuples():
        print(row)
        bound_list = ["upper" , "lower"]
        for bound in bound_list:
            cost = add_bounds_to_df(ps.costs, "Cost")
            if bound == "upper":
                cost.loc[row.Index, "Cost"] = row.Upper_Bound
                icers, opt_strat, opt_icers = generate_icers_all_genes(utility, cost)
                print(icers)
                MLH1_upper_icer.append(icers[0])
                MSH2_upper_icer.append(icers[1])
                MSH6_upper_icer.append(icers[2])
                PMS2_upper_icer.append(icers[3])
                MLH1_opt_upper_strat.append(opt_strat[0])
                MSH2_opt_upper_strat.append(opt_strat[1])
                MSH6_opt_upper_strat.append(opt_strat[2])
                PMS2_opt_upper_strat.append(opt_strat[3])
                MLH1_opt_upper_icer.append(opt_icers[0])
                MSH2_opt_upper_icer.append(opt_icers[1])
                MSH6_opt_upper_icer.append(opt_icers[2])
                PMS2_opt_upper_icer.append(opt_icers[3])
            else:
                cost.loc[row.Index, "Cost"] = row.Lower_Bound
                icers, opt_strat, opt_icers = generate_icers_all_genes(utility, cost)
                print(icers)
                MLH1_lower_icer.append(icers[0])
                MSH2_lower_icer.append(icers[1])
                MSH6_lower_icer.append(icers[2])
                PMS2_lower_icer.append(icers[3])
                MLH1_opt_lower_strat.append(opt_strat[0])
                MSH2_opt_lower_strat.append(opt_strat[1])
                MSH6_opt_lower_strat.append(opt_strat[2])
                PMS2_opt_lower_strat.append(opt_strat[3])
                MLH1_opt_lower_icer.append(opt_icers[0])
                MSH2_opt_lower_icer.append(opt_icers[1])
                MSH6_opt_lower_icer.append(opt_icers[2])
                PMS2_opt_lower_icer.append(opt_icers[3])
    cost.loc[:, "Upper_MLH1_ICER"] = MLH1_upper_icer
    cost.loc[:, "Lower_MLH1_ICER"] = MLH1_lower_icer
    cost.loc[:, "Upper_MSH2_ICER"] = MSH2_upper_icer
    cost.loc[:, "Lower_MSH2_ICER"] = MSH2_lower_icer
    cost.loc[:, "Upper_MSH6_ICER"] = MSH6_upper_icer
    cost.loc[:, "Lower_MSH6_ICER"] = MSH6_lower_icer
    cost.loc[:, "Upper_PMS2_ICER"] = PMS2_upper_icer
    cost.loc[:, "Lower_PMS2_ICER"] = PMS2_lower_icer
    cost.loc[:, "Upper_MLH1_OPT_ICER"] = MLH1_opt_upper_icer
    cost.loc[:, "Lower_MLH1_OPT_ICER"] = MLH1_opt_lower_icer
    cost.loc[:, "Upper_MSH2_OPT_ICER"] = MSH2_opt_upper_icer
    cost.loc[:, "Lower_MSH2_OPT_ICER"] = MSH2_opt_lower_icer
    cost.loc[:, "Upper_MSH6_OPT_ICER"] = MSH6_opt_upper_icer
    cost.loc[:, "Lower_MSH6_OPT_ICER"] = MSH6_opt_lower_icer
    cost.loc[:, "Upper_PMS2_OPT_ICER"] = PMS2_opt_upper_icer
    cost.loc[:, "Lower_PMS2_OPT_ICER"] = PMS2_opt_lower_icer
    cost.loc[:, "Upper_MLH1_OPT_STRAT"] = MLH1_opt_upper_strat
    cost.loc[:, "Lower_MLH1_OPT_STRAT"] = MLH1_opt_lower_strat
    cost.loc[:, "Upper_MSH2_OPT_STRAT"] = MSH2_opt_upper_strat
    cost.loc[:, "Lower_MSH2_OPT_STRAT"] = MSH2_opt_lower_strat
    cost.loc[:, "Upper_MSH6_OPT_STRAT"] = MSH6_opt_upper_strat
    cost.loc[:, "Lower_MSH6_OPT_STRAT"] = MSH6_opt_lower_strat
    cost.loc[:, "Upper_PMS2_OPT_STRAT"] = PMS2_opt_upper_strat
    cost.loc[:, "Lower_PMS2_OPT_STRAT"] = PMS2_opt_lower_strat
    
    cost.to_csv("cost_owsa_icer_table_new.csv")
    return cost

def run_owsa_CEA_csy():
    utility = ps.utils_m.copy()
#    print(type(utility))
    cost = ps.costs.copy()
    columns  = ["du_csy", "du_csy_comp"]
    icer_df = pd.DataFrame(columns, columns=["Parameter"])
#   icer lists (make all code below more efficient)
    MLH1_upper_icer = []
    MLH1_lower_icer = []
    MSH2_upper_icer = []
    MSH2_lower_icer = []
    MSH6_upper_icer = []
    MSH6_lower_icer = []
    PMS2_upper_icer = []
    PMS2_lower_icer = []
    for col in columns:
        print(col)
        bound_list = ["upper" , "lower"]
        for bound in bound_list:
            if col == "du_csy":
                csy = owsa_range(ps.csy_disutil)
                if bound == "upper":
#                    cost.loc[, "Cost"] = row.Upper_Bound
                    icers = generate_icers_all_genes_csy(utility, cost, csy[0], ps.compl_disutil, overwrite=True)
                    MLH1_upper_icer.append(icers[0])
                    MSH2_upper_icer.append(icers[1])
                    MSH6_upper_icer.append(icers[2])
                    PMS2_upper_icer.append(icers[3])
                else:
#                    cost.loc[row.Index, "Cost"] = row.Lower_Bound
                    icers = generate_icers_all_genes_csy(utility, cost, csy[1], ps.compl_disutil, overwrite=True)
                    MLH1_lower_icer.append(icers[0])
                    MSH2_lower_icer.append(icers[1])
                    MSH6_lower_icer.append(icers[2])
                    PMS2_lower_icer.append(icers[3])
            else:
                comp_csy = owsa_range(ps.compl_disutil)
                if bound == "upper":
#                    cost.loc[row.Index, "Cost"] = row.Upper_Bound
                    icers = generate_icers_all_genes_csy(utility, cost, ps.csy_disutil, comp_csy[0], overwrite=True)
                    MLH1_upper_icer.append(icers[0])
                    MSH2_upper_icer.append(icers[1])
                    MSH6_upper_icer.append(icers[2])
                    PMS2_upper_icer.append(icers[3])
                else:
#                    cost.loc[row.Index, "Cost"] = row.Lower_Bound
                    icers = generate_icers_all_genes_csy(utility, cost, ps.csy_disutil, comp_csy[1], overwrite=True)
                    MLH1_lower_icer.append(icers[0])
                    MSH2_lower_icer.append(icers[1])
                    MSH6_lower_icer.append(icers[2])
                    PMS2_lower_icer.append(icers[3])
    icer_df.loc[:, "Upper_MLH1_ICER"] = MLH1_upper_icer
    icer_df.loc[:, "Lower_MLH1_ICER"] = MLH1_lower_icer
    icer_df.loc[:, "Upper_MSH2_ICER"] = MSH2_upper_icer
    icer_df.loc[:, "Lower_MSH2_ICER"] = MSH2_lower_icer
    icer_df.loc[:, "Upper_MSH6_ICER"] = MSH6_upper_icer
    icer_df.loc[:, "Lower_MSH6_ICER"] = MSH6_lower_icer
    icer_df.loc[:, "Upper_PMS2_ICER"] = PMS2_upper_icer
    icer_df.loc[:, "Lower_PMS2_ICER"] = PMS2_lower_icer
    icer_df.to_csv("csy_owsa_icer_table.csv")
    return cost



def run_owsa_CEA_util():
    columns = ps.utils_m.columns[6:16]
    utility = add_bounds_to_df(ps.utils_m, columns)
#    print(utility.columns)
#    print(type(utility))
    cost = ps.costs.copy()
    icer_df = pd.DataFrame(columns, columns=["Parameter"])
#   icer lists (make all code below more efficient)
    MLH1_upper_icer = []
    MLH1_lower_icer = []
    MSH2_upper_icer = []
    MSH2_lower_icer = []
    MSH6_upper_icer = []
    MSH6_lower_icer = []
    PMS2_upper_icer = []
    PMS2_lower_icer = []
    for col in columns:
        print(col)
        bound_list = ["upper" , "lower"]
        for bound in bound_list:
            utility = add_bounds_prob_to_df(ps.utils_m, columns)
            print(utility)
            if bound == "upper":
                print(utility.loc[:, col + " UB"])
                utility.loc[:, col] = utility.loc[:, col + " UB"]
                icers = generate_icers_all_genes(utility, cost, overwrite=True)
                MLH1_upper_icer.append(icers[0])
                MSH2_upper_icer.append(icers[1])
                MSH6_upper_icer.append(icers[2])
                PMS2_upper_icer.append(icers[3])
            else:
                utility.loc[:, col] = utility.loc[:, col + " LB"]
                icers = generate_icers_all_genes(utility, cost, overwrite=True)
                MLH1_lower_icer.append(icers[0])
                MSH2_lower_icer.append(icers[1])
                MSH6_lower_icer.append(icers[2])
                PMS2_lower_icer.append(icers[3])
    icer_df.loc[:, "Upper_MLH1_ICER"] = MLH1_upper_icer
    icer_df.loc[:, "Lower_MLH1_ICER"] = MLH1_lower_icer
    icer_df.loc[:, "Upper_MSH2_ICER"] = MSH2_upper_icer
    icer_df.loc[:, "Lower_MSH2_ICER"] = MSH2_lower_icer
    icer_df.loc[:, "Upper_MSH6_ICER"] = MSH6_upper_icer
    icer_df.loc[:, "Lower_MSH6_ICER"] = MSH6_lower_icer
    icer_df.loc[:, "Upper_PMS2_ICER"] = PMS2_upper_icer
    icer_df.loc[:, "Lower_PMS2_ICER"] = PMS2_lower_icer
#    icer_df.to_csv("utility_owsa_icer_table_v2.csv")
    return icer_df

def run_owsa_CEA_prob():
    utility = ps.utils_m.copy()
#    print(type(utility))
    cost = ps.costs.copy()
#   icer lists (make all code below more efficient)
    MLH1_upper_icer = []
    MLH1_lower_icer = []
    MSH2_upper_icer = []
    MSH2_lower_icer = []
    MSH6_upper_icer = []
    MSH6_lower_icer = []
    PMS2_upper_icer = []
    PMS2_lower_icer = []
    icers = generate_icers_all_genes(utility, cost)
    MLH1_upper_icer.append(icers[0])
    MSH2_upper_icer.append(icers[1])
    MSH6_upper_icer.append(icers[2])
    PMS2_upper_icer.append(icers[3])
    icer_df.loc[:, "Upper_MLH1_ICER"] = MLH1_upper_icer
#    icer_df.loc[:, "Lower_MLH1_ICER"] = MLH1_lower_icer
    icer_df.loc[:, "Upper_MSH2_ICER"] = MSH2_upper_icer
#    icer_df.loc[:, "Lower_MSH2_ICER"] = MSH2_lower_icer
    icer_df.loc[:, "Upper_MSH6_ICER"] = MSH6_upper_icer
#    icer_df.loc[:, "Lower_MSH6_ICER"] = MSH6_lower_icer
    icer_df.loc[:, "Upper_PMS2_ICER"] = PMS2_upper_icer
#    icer_df.loc[:, "Lower_PMS2_ICER"] = PMS2_lower_icer
#    icer_df.to_csv("probability_icer_table_v2.csv")
    return icer_df


#def run_owsa_CEA():
#    for row in ps.costs:
        #graph_eff_frontiers(df)
#generate_ICER_with_plots_fancy(pd.read_csv('all_ce_results_male_same_RRs.csv'))
#r = run_owsa_CEA()
#print(r)
#
#strat_list = r.loc[:, 'gene'] + ' ' + r.loc[:, 'Strategy']
