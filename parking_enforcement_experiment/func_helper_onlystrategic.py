import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import tikzplotlib

def compute_value_onlystrategic(df_space, permit_dict, param_):
    
    '''
    Computes the value corresponding to each location and permit type in dataframe
    df_space. param_specifies fraction of people that engage in strategic behavior.
    Output: df_space with value information
    '''
    for permit, value in permit_dict.items():
        new_column_name = f'Value of {permit}'
        df_space[new_column_name] = df_space[permit] * value * param_

    return df_space

def compute_d_val_onlystrategic(df_space, permit_dict, threshold_prob):

    '''
    Populate the threshold probability values for each permit type and lot based on theory
    '''
    for permit, value in permit_dict.items():
        column_name = f'd val {permit}'
        df_space[column_name] = threshold_prob[value]

    return df_space

'''
Genderate the list of all types for each location
'''
def generate_types(row):
    '''
    Generates the pairs of the value and threshold probabilities corresponding to each lot
    for the different user types in ascending order of the threshold probabilities.
    Input: Row of df_space dataframe and a mapping from d to threshold prob
    Note that each row will consist of some 0 and non-zero values
    For all non-zero values populate the pairs of the (value, threshold prob)
    Output: Sorted list of pairs in increasing order of threshold prob
    '''
    # List of (value, threshold probability) pairs
    pairs = []
    
    # Iterate over each permit type
    for permit in permit_types:
        value_col = f'Value of {permit}'
        d_col = f'd val {permit}'
        
        # Check if the value is not zero
        if row[value_col] > 0:
            pairs.append((row[value_col], row[d_col]))

    # Sort pairs by increasing d value
    pairs.sort(key=lambda x: x[1])
    
    return pairs

def plot_welfare_num_strategic(strategic_arr, existing_welfare, greedy_welfare, random_welfare):

    fig, ax = plt.subplots()
    ax.plot(strategic_arr, existing_welfare)
    ax.plot(strategic_arr, greedy_welfare)
    ax.plot(strategic_arr, random_welfare)
    ax.legend(['Actual', 'Greedy', 'Random'])
    ax.set_xlabel('Proportion of Strategic Users')
    ax.set_ylabel('Total Welfare')

def plot_welfare_frac_num_strategic(strategic_arr, existing_welfare, greedy_welfare, random_welfare, max_val):

    greedy_welfare_total = []
    existing_welfare_total = []
    random_welfare_total = []
    for idx, frac_strategic_ in enumerate(strategic_arr):
        baseline_permit_earnings = max_val * (1-frac_strategic_)
        greedy_welfare_total.append((greedy_welfare[idx] + baseline_permit_earnings)/max_val)
        random_welfare_total.append((random_welfare[idx] + baseline_permit_earnings)/max_val)
        existing_welfare_total.append((existing_welfare[idx] + baseline_permit_earnings)/max_val)

    print(greedy_welfare_total[2], existing_welfare_total[2])
    fig, ax = plt.subplots()
    ax.plot(strategic_arr, existing_welfare_total)
    ax.plot(strategic_arr, greedy_welfare_total)
    ax.plot(strategic_arr, random_welfare_total)
    ax.plot(strategic_arr, [1 - i for i in strategic_arr])
    #ax.legend(['Status Quo', 'Our Algorithm', 'Random', 'No Enforcement'])
    ax.set_xlabel('Proportion of Strategic Users')
    ax.set_ylabel('Fraction of Total Permit Revenue')
    tikzplotlib.save("payoff_plot_frac_permit_rev_counterfactual1.tex")