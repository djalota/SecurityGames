import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_ub_points(pairs):
    '''
    Compute the list points corresponding to the upper bound of the welfare function
    for a given location
    Input: List of pairs of (value, threshold prob) sorted in increasing order of threshold prob
    Output: List of pairs of (upper bound value, threshold prob)
    '''
    ub_pairs = [(0, 0)]
    sum_v = 0
    for x, thresh in pairs:
        frac_val = thresh * (sum([j_val[0] for j_val in pairs if j_val[1] > thresh]))
        sum_v += x
        ub_val = sum_v + frac_val  # Update ub_val
        ub_pairs.append((ub_val, thresh))
        
    return ub_pairs

def compute_lb_points(pairs):
    '''
    Compute the list of points corresponding to the lower bound of the welfare function
    for a given location
    Input: List of pairs of (value, threshold prob) sorted in increasing order of threshold prob
    Output: List of pairs of (lower bound value, threshold prob)
    '''
    
    ub_lb_points = []
    sum_v = 0
    for x, thresh in pairs:
        frac_val_greater = thresh * (sum([j_val[0] for j_val in pairs if j_val[1] > thresh]))
        frac_val_eq_greater = thresh * (sum([j_val[0] for j_val in pairs if j_val[1] >= thresh]))
        add_store = sum([j_val[0] for j_val in pairs if j_val[1] >= thresh])
        lb_val = sum_v #+ frac_val_eq_greater #compute lb_val
        sum_v += x                           #compute sum of computed values till now
        ub_val = sum_v + frac_val_greater    #compute ub_val
        ub_lb_points.append((ub_val, thresh, lb_val, add_store))
        
    return ub_lb_points

def upper_convex_hull(points):
    # Sort the points lexicographically (tuples are compared element by element)
    points = sorted(points)

    # Build the upper hull
    upper = []
    for p in points:
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return upper

def cross_product(o, a, b):
    # This function will calculate the cross product of vector OA and OB
    # A positive cross product indicates a counter-clockwise turn, 0 indicates a collinear point, and negative indicates a clockwise turn
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def compute_location_welfare(amt_allocated, lot_data):
    '''
    Given an allocation to a particular lot, compute the corresponding welfare at that location
    Input: Allocation amount and corresponding lot information, inlcuding lot id and welfare_func data
    Welfare func has the different points corresponding to the welfare function
    Output: Welfare accrued at that location given the amt_allocated
    '''
    
    tuples = lot_data.copy() #[1]

    if amt_allocated < tuples[0][1]:                       #If amt allocated is less than lowest resource req for location, then accrue fraction of value
        return amt_allocated * tuples[0][-1]
    
    for i in range(1, len(tuples)):
        
        if amt_allocated == tuples[i][1]:                  #If amt allocated is exactly equal to thresholds, then just select the UB value
            return tuples[i][0]
        
        if tuples[i-1][1] <= amt_allocated < tuples[i][1]:  #If amt allocated is between two thresholds, interpolate
            return tuples[i-1][0] #+ (amt_allocated - tuples[i-1][1]) * (tuples[i][-1])
    
    if amt_allocated >= tuples[-1][1]:                     #If amt allocated is above max threshold, then set UB of last value
        return tuples[-1][0]    

def compute_total_welfare(allocation, results2):
    '''
    Given an allocation, compute the corresponding welfare
    Input: An allocation dictionary with keys as the lot Id and the values as
    the amount of resources allocated to that lot
    Output: Welfare corresponding to allocation
    '''
    total_welfare = 0
    for lot, data in results2:
        total_welfare += compute_location_welfare(allocation[lot], data)
    
    return total_welfare

def plot_func(actual_welfare, final_obj_val, max_val):

    '''
    Plot the bar graph of the performance of ALG to Actual
    Actual performance = actual_welfare/max_val
    Greedy Performance = final_obj_val/max_val
    '''

    # Define the width of the bars
    bar_width = 0.35

    # Plotting both series of data
    fig, ax = plt.subplots()
    bars1 = ax.bar(0, actual_welfare/max_val, bar_width, label='Actual')
    bars2 = ax.bar(1, final_obj_val/max_val, bar_width, label='ALG')

    # Adding labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Fraction of Total Welfare Accrued')
    ax.set_title('Comparison of Actual and ALG Welfare by Month')
    ax.legend()

    # Show the plot
    plt.show()