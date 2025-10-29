# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:34:58 2019

@author: mai2125
"""
import probability_functions as pf
import numpy as np
import lynch_presets as ps
import pandas as pd
import lynch_simulator as sim
import probability_functions as pf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import lynch_icer as ic
import gender as g
import data_manipulation as dm
import multiprocessing as mp
import csv
#PSA

# =============================================================================
# PLAN OF ATTACK
# 1. Get costs/utilities/parameters in df format
# 2. Convert them using the appropriate distributions
# 3. Reinsert the new values into code for ICER/simulation
# 
# Costs: gamma
# Probabilities: beta
# Utilities: uniform 
# =============================================================================


#utilities and costs for all arms
#def unif_dist(df):
#    for i in range(df.shape[1]):
#        col_val = df.values[:, i]
#        if col_val[0] == 0:
#            continue
#        else:
#            val_list = []
#            first_val = np.random.uniform(col_val[0]*.5, col_val[0])
#            val_list.append(first_val)
#            for j in range(len(col_val)-1):
#                if col_val[j + 1] != col_val[j]:
#                    x = np.random.uniform(col_val[j]*.5, col_val[j])
#                elif j == len(col_val)-1:
#                    x = np.random.uniform(col_val[j]*.5, col_val[j])
#                val_list.append(x)
#            df.iloc[:, i] = val_list
#    return df
#
#
#def gamma_dist(old_list):
#    k_list, th_list = [], []        
#    for x in old_list:
#        k, th = pf.gamma_dist(x, x*.1)
#        k_list.append(k)
#        th_list.append(th)
#    new_list = [np.random.gamma(k, th) for k, th in zip(k_list, th_list)]
#    return new_list
#
#
#def beta_dist(old_list):
#    a_list, b_list = [], []
#    for x in old_list:
#        a, b = pf.beta_dist(x, x*.1)
#        a_list.append(a)
#        b_list.append(b)
#    new_list = [np.random.beta(a, b) for a, b in zip(a_list, b_list)]
#    return new_list


def icer_plot(inc_cost_list, inc_eff_list, pt_color, gene):
    WTP = 100000

    plt.scatter(inc_eff_list, inc_cost_list, c=pt_color, 
                s=5, alpha=0.6, facecolors = None)
    lower = plt.axis()[0]
    upper = plt.axis()[1]
    
    plt.plot([lower, upper], np.multiply([lower, upper], WTP), "r--", alpha=0.5, label='WTP threshold')
    
    plt.xlabel("Incremental Effectiveness")
    plt.ylabel("Incremental Cost (USD)")
#    plt.xlim(-0.2, 0.2)
    if gene == "MLH1":
        plt.ylim(3500, 8000)
        plt.title("MLH1 PSA Q1Y, Start Age: 25")
        plt.savefig("MLH1_PSA_v3.png")
    elif gene == "MSH2":
        plt.ylim(1000, 3500)
        plt.title("MSH2 PSA Q2Y, Start Age: 25")
        plt.savefig("MSH2_PSA_v3.png")
    elif gene == "MSH6":
#        plt.ylim(500, 2000)
        plt.title("MSH6 PSA Q3Y, Start Age: 40")
        plt.savefig("MSH6_PSA_v3.png")
    elif gene == "PMS2":
        plt.title("PMS2 PSA Q3Y, Start Age: 40")
        plt.savefig("PMS2_PSA_v3.png")
        
    plt.show()
    return


def launch_plot(inc_cost_list, inc_eff_list, pt_color):
    WTP = 100000
    lower = plt.axis()[0]
    upper = plt.axis()[1]
    plt.plot([lower, upper], np.multiply([lower, upper], WTP), "r--", alpha=0.5, label='WTP threshold')
    plt.legend(('WTP threshold', "PAC vs BSC", "MSI-H: PEM / MSS: PAC vs PAC"), loc='upper left', frameon=True)
    plt.xlabel("Incremental Effectiveness")
    plt.ylabel("Incremental Cost (USD)")
    plt.xlim(-0.3,0.7)
    plt.ylim(-25000,150000)
    plt.grid(False)
    plt.show()
    return

def count_strat(result_df, strat_dict, WTP):
# counts the amounts of times the strategy is cost-effective
#    for gene in result_df.loc[:, 'gene']:
    gene = result_df.loc[0, "gene"]
    for age_csy in result_df.loc[:, "Strategy"]:
        strat = gene + ' ' + age_csy
        if strat in strat_dict.keys():
            row = result_df.loc[result_df['gene'].isin([gene]) & 
                                    result_df['Strategy'].isin([age_csy])].index[0]
            WTP_cost = result_df.loc[row, 'QALYs'] * WTP
#               print(WTP_cost)
#                print(row)
            if result_df.loc[row, 'cost'] <= WTP_cost:
                strat_dict[strat] += 1
        return strat_dict


def WTP_check(df, strat_dict, WTP):
    gene = df.loc[0, "gene"]
    count = 0
#    print(df)
    while df.iloc[count, 12] > WTP:
        count += 1
        continue
    strat = df.iloc[count, 0] + ' ' + df.iloc[count, 1]
    strat_dict[strat] += 1
    return strat_dict

    
def count_best_strat(df, strat_dict, WTP):
    QALYs_sort = df.sort_values("QALYs", ascending=False)
    strat_dict = WTP_check(QALYs_sort, strat_dict, WTP)
    
    return strat_dict


def percent_ce(strat_dict, run_times):
# turns the count into a percentage of cost-effective for all runs
    strat_perc_dict = strat_dict.copy()
    for key in strat_dict.keys():
        percentage = strat_dict[key]/run_times
        print(str(percentage) + "% of iterations were greater than the WTP threshold for " + key)
        strat_perc_dict[key] = percentage
        
    return strat_perc_dict
        

def run_psa(run_times, strat_list, WTP):
# inputs are run_times and strat_list is from the strategies of df of run_CEA
    zeros = [0] * len(strat_list)
    strat_count = dict(zip(strat_list, zeros))
    print("in run_psa...")
    for i in range(run_times):
        print(i)
        df = ic.run_CEA()
        for gene in ps.genes:
            print('generating efficiency frontier for', gene)
        this_df = df[df['gene'] == 'MSH6']
        this_df = this_df.reset_index(drop = True)
        icers = ic.get_icers(this_df)
        strat_count = count_strat(icers, strat_count, 'MSH6', WTP)
    strat_perc_dict = percent_ce(strat_count, run_times)
    return strat_perc_dict

def raw_ce_psa(this_df):
    '''
    Gives the raw incremental cost and effectiveness for the strategy with the 
    highest QALYs for a given run of the PSA
    '''
    # eliminates strategies above WTP
    this_df = this_df[this_df.icers <= ps.WTP]
    # checks if this_df has 2 or more strategies
    if this_df.shape[0] >= 2: 
        inc_cost = this_df["cost"].values[-1] - this_df["cost"].values[-2]
        inc_eff = this_df["QALYs"].values[-1] - this_df["QALYs"].values[-2]
    else:
        inc_cost = 0
        inc_eff = .05
    dom_strat = this_df["Strategy"].values[-1]
    
        
    return inc_cost, inc_eff, dom_strat

def gene_dict_list():
    '''
    Creates a list of dictionaries for the results of the PSA to be stored in
    '''
    gene_dicts = []
    for gene in ps.genes:
        result_dict = {"dom_strat":[], "inc_cost":[], "inc_eff":[]}
        gene_dicts.append(result_dict)
    return gene_dicts

def run_best_psa(run_times, strat_list, WTP):
# inputs are run_times and strat_list is from the strategies of df of run_CEA
    gene_dicts = gene_dict_list()
    zeros = [0] * len(strat_list)
    strat_count = dict(zip(strat_list, zeros))
#    print(strat_count)
    print("in run_psa...")
    for i in range(run_times):
        print(i)
        df = ic.run_CEA_PSA()
        j = 0
        for gene in ps.genes:
#            print('generating efficiency frontier for', g
            this_df = df[j]
            strat_count = count_best_strat(this_df, strat_count, WTP)
            inc_cost, inc_eff, dom_strat = raw_ce_psa(this_df)
            gene_dicts[j]["dom_strat"].append(dom_strat)
            gene_dicts[j]["inc_eff"].append(inc_cost)
            gene_dicts[j]["inc_cost"].append(inc_eff)
            j += 1
#            print(strat_count)
    strat_perc_dict = percent_ce(strat_count, run_times)
    return strat_perc_dict, gene_dicts


def comp_psa_gene(run_times, gene):
    inc_cost_list = []
    inc_eff_list = []
    if gene == "MLH1":
        comp_list = ps.MLH1_psa_strat
    elif gene == "MSH2":
        comp_list = ps.MSH2_psa_strat
    elif gene == "MSH6":
        comp_list = ps.MSH6_psa_strat
    elif gene == "PMS2":
        comp_list = ps.PMS2_psa_strat
    else:
        print("ERROR: Invalid gene entry")
    
    count = 0
    for i in range(run_times):
        print(i)
        df = ic.run_CEA()
        this_df = df[df['gene'] == gene]
        this_df = dm.selection(this_df, comp_list, "Strategy").reset_index(drop=True)
#        print(this_df).;
#        print(this_df["QALYs"])
        for j in this_df["cost"].values:
            inc_cost = this_df["cost"].values[0] - this_df["cost"].values[1]
            inc_cost_list.append(inc_cost)
        for k in this_df["QALYs"].values:
            inc_eff = this_df["QALYs"].values[0] - this_df["QALYs"].values[1]
            inc_eff_list.append(inc_eff)
#        if inc_cost/inc_eff > 100000:
    return inc_cost_list, inc_eff_list


def comp_strat_list(strat_list):
    c = []
    for gene in ps.genes:
        for strat in strat_list:
            n = gene + ' ' + strat
            c.append(n)
    return c


def run_comp_psa(run_times):
#   Outputs a scatter plot for each strategy comparison for each gene
    for gene in ps.genes:
        inc_cost_list, inc_eff_list = comp_psa_gene(run_times, gene)
#    print(inc_cost_list)
        icer_plot(inc_cost_list, inc_eff_list, 'g', gene)
        
#        launch_plot()
    return
        

def save_dict(dictionary):
    '''
    Saves the winning strategy psa results
    '''
    w = csv.writer(open(ps.psa/"psa_results_dictionary.csv", "w"))
    for key, val in dictionary.items():
        w.writerow([key, val])
    return
            

def save_gene_dicts(gene_dicts):
    '''
    Saves the ICER for each psa run
    '''
    
    for i in range(len(gene_dicts)):
        df = pd.DataFrame(gene_dicts[i])
        filename = "psa_results_icer_scatter_" + ps.genes[i] + ".csv"
        df.to_csv(ps.psa/filename)
    
    return

def load_strat_dict(filename):
    strat_df = pd.read_csv(filename).dropna()
    strat_df = strat_df.set_index("Strategy")
    strat_dict = strat_df.to_dict()
    return strat_dict
    



#strat_dict = run_psa(1000, ps.strat_list, 100000) 
# make the result into a graph

def psa_plot_h(strat_dict):
    
    dom_dict = {x:y for x, y in strat_dict.items() if y != 0}
    plt.title("PSA Optimal Strategy Percentage")
    plt.ylabel("Strategy")
    plt.xlabel("% of runs optimal")
    plt.xlim(0, 1.1)
    plt.barh(*zip(*dom_dict.items()))
    file_name = 'PSA_all_combined_gender_h.png'
    plt.savefig(ps.psa/file_name, dpi=200)
    plt.show()
    return

def psa_plot_gene_h(strat_dict):
    
    dom_dict = {x:y for x, y in strat_dict.items() if y != 0}
    fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2, figsize = (17, 14))
    plt_list = [plt1, plt2, plt3, plt4]
#    plt.suptitle('Optimal Strategy Percentage of PSA',
#                 fontsize = 20, y = 0.94)
    i = 0
    for gene in ps.genes:
        gene_dict = {x.strip(gene)[1:]:y for x, y in dom_dict.items() if gene in x}
        print(gene_dict)
        plt_list[i].set_title(gene, fontsize=14)
        plt_list[i].set_xlabel("Strategy", fontsize=14)
        plt_list[i].set_ylabel("% of runs optimal", fontsize=14)
        if gene == "MLH1":
            c = 'g'
        elif gene == "MSH2":
            c = 'r'
        elif gene == "MSH6":
            c = 'k'
        elif gene == "PMS2":
            c = 'm'
        for item in (plt_list[i].get_xticklabels() + plt_list[i].get_yticklabels()):
            item.set_fontsize(12)
        plt_list[i].bar(*zip(*gene_dict.items()), color=c)
        plt_list[i].set_ylim(0, 1.1)
        
        i += 1
    file_name = 'PSA_combined_gender_composite_1.png'
    plt.savefig(ps.psa/file_name, dpi=500)
    plt.show()
    return

def psa_plot_from_filename(filename):
    strat_dict = load_strat_dict(filename)["Value"]
    psa_plot_gene_h(strat_dict)    
    return

#def psa_subplot_genes():
# =============================================================================
# def psa_plot(strat_dict):
#     
#     dom_dict = {x:y for x, y in strat_dict.items() if y != 0}
#     plt.title("PSA Cost-Effective Percentage")
#     plt.xlabel("Strategy")
#     plt.ylabel("% of runs best strategy")
#     plt.ylim(0, 1)
#     plt.bar(*zip(*dom_dict.items()))
#     file_name = 'PSA_all_combined_gender.png'
#     plt.savefig(ps.psa/file_name, dpi = 200)
#     plt.show()
#     return
# 
# def psa_plot_gene(strat_dict, gene):
#     
#     dom_dict = {x:y for x, y in strat_dict.items() if y != 0}
#     gene_dict ={x:y for x, y in dom_dict.items() if gene in x}
#     plt.title("PSA Cost-Effective Percentage")
#     plt.xlabel("Strategy")
#     plt.ylabel("% of runs best strategy")
#     if gene == "MLH1":
#         c = 'g'
#     elif gene == "MSH2":
#         c = 'r'
#     elif gene == "MSH6":
#         c = 'k'
#     elif gene == "PMS2":
#         c = 'm'
#     plt.ylim(0, 1)
#     plt.bar(*zip(*gene_dict.items()), color=c)
#     file_name = 'PSA_' + gene + '_combined_gender.png'
#     plt.savefig(ps.psa/file_name, dpi = 200)
#     plt.show()
#     return
# =============================================================================
    

def psa():
    strat_dict = run_psa(1000, ps.strat_list, 100000)
    icer_plot(strat_dict)
    return


def psa_best():
    
    strat_dict, gene_dicts = run_best_psa(1000, ps.strat_list, 100000)
    save_gene_dicts(gene_dicts)
    save_dict(strat_dict)
    psa_plot_h(strat_dict)
   
    psa_plot_gene_h(strat_dict)
    return
        
#run_comp_psa(1000)





        
        
        
        
        
        
        
        
        