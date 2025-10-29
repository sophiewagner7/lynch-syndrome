#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:09:39 2020

@author: mai2125
"""

def calibrate_markov(arm, run_time, morb_table, target_value, ans_num, *args):
    '''
    Loops over the markov model making process, randomizing select parameters
    until the p-value of the output of the markov goes below the target value and
    saves the number of solutions specified
    
    
    Input: variable of the arm class, run time of the model, relevant morbidity
    table, the target value that the p-value needs to be under/equal to, and 
    if available special connectivity dict for a range of cycles
    
    Output: possible t_matrices and D_matrices that fit the GOF parameters
    '''    
    # initialize answer list
    answers = []
    # gets pw slopes and nodes from KM curves (make into one function)
    if arm.PFS_target:
        slopes_PFS, nodes_PFS = lf.extract_KM_rates(arm.PFS_target, 
                                                arm.PFS_nodes[0], arm.PFS_nodes[1])
#    KM_time_PFS, KM_PFS = dm.csv_to_lists(arm.PFS_target)
    if arm.OS_target:
        slopes_OS, nodes_OS = lf.extract_KM_rates(arm.OS_target,
                                              arm.OS_nodes[0], arm.OS_nodes[1])
#    KM_times_OS, KM_OS = dm.csv_to_lists(arm.OS_target)
    # loops over markov creation until an adequete p value is found
    while len(answers) < ans_num:
        # initialize p values
        print(len(answers))
        if arm.OS_target:
            p_value_OS = target_value +1
        if arm.PFS_target:
            p_value_PFS = target_value +1
        # runs markov until a t_matrix is created that fits the expected values
        while (p_value_OS >= target_value) or (p_value_PFS >= target_value):
            A = deepcopy(arm)
            A.params_dict = randomize_params(A.params_dict, A.calib_params, .98)
            t_matrix, D_matrix = run_markov(A, run_time, morb_table, *args)
#            print(D_matrix)
            if arm.OS_target:
                OS_sum = get_survival(D_matrix, ps.death_states)
                p_value_OS = gof(1-OS_sum, nodes_OS, slopes_OS)
                
            if arm.PFS_target:
                PFS_sum = get_survival(D_matrix, ps.disease_states)
                p_value_PFS = gof(1-PFS_sum, nodes_PFS, slopes_PFS)
                
            print("p_value OS: " + str(p_value_OS))
            print("p_value PFS: " + str(p_value_PFS))
            # save params_dict
        answers.append([t_matrix, D_matrix, A.params_dict])
        
    return answers
        