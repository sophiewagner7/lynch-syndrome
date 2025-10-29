from scipy import optimize
from scipy import stats
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_manipulation as dm
import lynch_presets as ps
from matplotlib.colors import LogNorm
import probability_functions as pf


def slope_list(rise, run):
    # Takes a list of x and y distances from their 
    # respective averages and returns a slope
    A = sum(np.multiply(rise, run))
    slope = A/sum(np.multiply(run, run))
    return slope
    
def intercept(x, y, m):
    # b = y - mx
    b = y - m * x
    return b

def mylinregress(x, y):
    # Takes x and y to produce a slope and intercept
    if (len(x) != len(y)):
        print("ERROR-- x_val and y_val have inconsistent lengths (in pw_linregress)")
        return
    
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    A_x = [i-x_avg for i in x]
    A_y = [j-y_avg for j in y]
    slope = slope_list(A_y, A_x)
    inter = intercept(x_avg, y_avg, slope)
    return slope, inter


#recode this later
def line_connect(x, y):
    
    x_bounds = [[x[i], x[i+1]] for i in range(len(x)-1)]
    y_bounds = [[y[i], y[i+1]] for i in range(len(y)-1)]

    return x_bounds, y_bounds

def pw_mylinregress(x_segments, y_segments):
    
    if (len(x_segments) != len(y_segments)):
        print("ERROR-- x_val and y_val have inconsistent lengths (in pw_linregress)")
        return
    

    slope = 0
    intercept = 0
    pw_slopes = list()
    pw_intercepts = list()
#    
##    for x_seg in x_segments, y_seg in y_segments:
    for x_seg, y_seg in zip(x_segments, y_segments):
        slope, intercept = mylinregress(x_seg, y_seg)
        pw_slopes.append(slope)
        pw_intercepts.append(intercept)
#    
    pw = pd.DataFrame([pw_slopes, pw_intercepts])
    return pw

def ensure_KM_bounds_consistency(x,y):
    
    if (x[0] < 0):
        x[0] = 0
        
    # check for y<0
        
    return x,y


def linpredict(x, m, b):
    p = m*x + b
    return p


def generate_segments(x,y, nodes):
    #exception
    if ( not (len(nodes) < len(x))): 
        print("Node length exceeds length of x")
    if ( not (len(x) == len(y))): #exception
        print("length of x does not match length of y")
    ind = 0
#    c = 0
    x_segments = list()
    y_segments = list()
    
    for i in range(len(nodes)-1):        
        x_segment = list()
        y_segment = list()
        for ind in range(len(x)):        
            if(x[ind] >= nodes[i] and x[ind] < nodes[i+1]):
                x_segment.append(x[ind])
                y_segment.append(y[ind])
#                print(ind)
        x_segments.append(x_segment)
        y_segments.append(y_segment)
        
    return x_segments, y_segments

# Takes a dataframe of slope and intercepts and plots them
# Assumes first row is slopes and second row is intercepts
def pw_plot(x, df, color):
    for i in df.columns:
        x[i] = np.array(x[i])
        pw_plot = plt.plot(x[i], df.loc[0, i]*x[i] + df.loc[1, i], color)
    return pw_plot


def pw_value(time, KM_df, nodes):
    slopes = KM_df.iloc[0, :]
    intercepts = KM_df.iloc[1, :]
    for n in range(len(nodes)-1):
        if pf.between(time, nodes[n], nodes[n+1]):
            value = linpredict(time, slopes[n], intercepts[n])
            return value
        elif time >= nodes[len(nodes)-1]:
            value = linpredict(time, slopes[len(nodes)-2], 
                                            intercepts[len(nodes)-2])
            return value

def pw_node_value(KM_df, nodes):
    node_values = [pw_value(node, KM_df, nodes) for node in nodes]
    node_values[0] = 0
    return node_values


def extract_values(filename):
    KM_df, nodes = extract_srvl(filename)
    values = [pw_value(t, KM_df, nodes) for t in range(int(nodes[-1]))]
    values[0] = 1
    return values



def r_squared(y, p):
    y_avg = sum(y)/len(y)
    A_y = []
    A_p = []
    for i in range(len(y)):
        d_y = y[i] - y_avg
        A_y.append(d_y)
        d_p = y[i] - p[i]
        A_p.append(d_p)
    ss_tot = sum(np.multiply(A_y, A_y))
    ss_res = sum(np.multiply(A_p, A_p))
    r_s = 1 - ss_res/ss_tot
    return r_s



def pw_r_squared(x, y, seg):
    df = pw_mylinregress(x, y)
    r_squared_list = []
    for i in range(len(x)):
        y_p = r_squared(y[i], linpredict(x[i], df.loc[0, i], df.loc[1, i]))
        r_squared_list.append(y_p)
    return


def extract_KM_rates(filename):
    t, value = dm.csv_to_lists(filename) 
    t = np.asarray(t)/ps.CYCLE_LENGTH 
    value = np.asarray(value)
    nodes = [25, 40, 60, 70]
    x, y = generate_segments(t, value, nodes)
    linfit_rates = pw_mylinregress(x, y)
    return linfit_rates, nodes

        
# extracts rates from excel sheets
def extract_rates(filename, sheet_name):
    t, value = dm.excel_to_lists(filename, sheet_name) 
    value = np.asarray(value)/100
    x, y = line_connect(t, value)
    linfit_rates = pw_mylinregress(x, y)
#    pw_plot(x, linfit_rates, 'g')
    return t, linfit_rates


def plot_linfit_KM(filename):
    t, value = dm.csv_to_lists(filename) 
    t = np.asarray(t)/ps.CYCLE_LENGTH 
    value = np.asarray(value)
    nodes = [25, 40, 60, 70]
    x, y = generate_segments(t, value, nodes)
    linfit_rates = pw_mylinregress(x, y)
    plt.plot(t, value, 'go')
    pw_plot(x, linfit_rates, 'r')
    return

def extract_srvl(filename):
    KM_t, KM_value = dm.csv_to_lists(filename) 
    KM_t = np.asarray(KM_t)/ps.CYCLE_LENGTH 
    KM_value = np.asarray(KM_value)
    x, y = [int(x) for x in input('Enter two nodes: ').split()]
    nodes = [min(KM_t), x, y, max(KM_t)+1]
    KM_x, KM_y = generate_segments(KM_t, KM_value, nodes)
    KM_linfit_rates = pw_mylinregress(KM_x, KM_y)
    return KM_linfit_rates, nodes


def plot_srvl(filename):
    KM_t, KM_value = dm.csv_to_lists(filename) 
    KM_t = np.asarray(KM_t)/ps.CYCLE_LENGTH 
    KM_value = np.asarray(KM_value)
    x, y = [int(x) for x in input('Enter two nodes: ').split()]
    nodes = [min(KM_t), x, y, max(KM_t)+1]
    KM_x, KM_y = generate_segments(KM_t, KM_value, nodes)
    KM_linfit_rates = pw_mylinregress(KM_x, KM_y)
    plt.plot(KM_t, KM_value, 'go')
    pw_plot(KM_x, KM_linfit_rates, 'r')
    return

