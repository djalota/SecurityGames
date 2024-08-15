import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import tikzplotlib

'''
Genderate the list of all types for each location
'''
def generate_types(row, percentiles):
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
        for p1, p2 in zip(percentiles[:-1], percentiles[1:]):
            value_col = f'Value of {permit} {int(p1 * 100)}-{int(p2 * 100)} percentile'
            d_col = f'd val {permit} {int(p1 * 100)}-{int(p2 * 100)} percentile'
            # Check if the value is not zero
            if row[value_col] > 0:
                pairs.append((row[value_col], row[d_col]))

    # Initialize a dictionary to hold the sums of the first entries, grouped by the second entry
    grouped = {}

    # Populate the dictionary
    for first, second in pairs:
        if second in grouped:
            grouped[second] += first
        else:
            grouped[second] = first

    # Convert the dictionary back to a list of tuples
    unique_pairs = [(value, key) for key, value in grouped.items()]
    
    # Sort pairs by increasing d value
    unique_pairs.sort(key=lambda x: x[1])
    
    return unique_pairs

# Function to get the midpoint from the percentile range in column name
def get_midpoint(col_name):
    # Check if 'percentile' is in the column name
    if 'percentile' in col_name:
        # Extract the range part of the name
        range_part = col_name.split(' ')[-2]
        # Extract the start and end of the range
        start, end = map(int, range_part.split('-'))
        # Calculate and return the midpoint
        return (start / 100.0 + end / 100.0) / 2
    return None

def plot_welfare_num_strategic(multiplier_arr, existing_welfare, greedy_welfare, other_greedy_welfare, random_welfare):

    fig, ax = plt.subplots()
    ax.plot(multiplier_arr, existing_welfare)
    ax.plot(multiplier_arr, greedy_welfare)
    ax.plot(multiplier_arr, other_greedy_welfare)
    ax.plot(multiplier_arr, random_welfare)
    #ax.legend(['Actual', 'Greedy', 'Other Greedy', 'Random'])
    ax.set_xlabel('Proportion of Strategic Users')
    ax.set_ylabel('Total Welfare')
    tikzplotlib.save("payoff_plot_permit_rev.tex")

def plot_welfare_frac_num_strategic(multiplier_arr, existing_welfare, greedy_welfare, other_greedy_welfare, random_welfare, max_val):

    greedy_welfare_total = []
    existing_welfare_total = []
    other_greedy_welfare_total = []
    random_welfare_total = []
    for idx in range(len(multiplier_arr)):
        greedy_welfare_total.append((greedy_welfare[idx])/max_val)
        existing_welfare_total.append((existing_welfare[idx])/max_val)
        other_greedy_welfare_total.append((other_greedy_welfare[idx])/max_val)
        random_welfare_total.append((random_welfare[idx])/max_val)

    fig, ax = plt.subplots()
    print(greedy_welfare_total[5], existing_welfare_total[5])
    ax.plot(multiplier_arr, existing_welfare_total)
    ax.plot(multiplier_arr, greedy_welfare_total)
    ax.plot(multiplier_arr, other_greedy_welfare_total)
    ax.plot(multiplier_arr, random_welfare_total)
    #ax.legend(['Status Quo', 'Our Algorithm', 'Our Algorithm (No Citation Data)', 'Random'])
    ax.set_xlabel('Citation Multiplier')
    ax.set_ylabel('Fraction of Total Permit Revenue')
    tikzplotlib.save("payoff_plot_frac_permit_rev.tex")