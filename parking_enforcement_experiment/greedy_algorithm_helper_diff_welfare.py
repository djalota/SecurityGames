import numpy as np
import pandas as pd
from helper_functions_diff_welfare import *
from utils import *

def generate_slope(key, points, i):
    '''
    Take in points and an index
    Output: list of slopes of the segments
    '''
    x1, y1 = points[i]
    x2, y2 = points[i + 1]
    dx = x2 - x1
    dy = y2 - y1
    if dy != 0:
        slope = dx / dy
    else:
        slope = float('inf')  # Handling vertical lines if any
        
    slope_tuple = (key, slope, dx, dy)
        
    return slope_tuple

def list_of_slopes(hulls):
    '''
    Take in the list of points corresponding to UB of welfare function for a set of lots
    Output: A sorted list of values of the slopes of the different segments
    '''
    # Create a list to store the slope and associated tuple information
    slope_list = []

    # Compute the differences and slopes
    for key, points in hulls.items():
        for i in range(len(points) - 1):
            slope_tuple = generate_slope(key, points, i)
            slope_list.append(slope_tuple)

    # Sort the list of slopes in descending order
    slope_list.sort(reverse = True, key=lambda x: x[1])
    
    return slope_list

#Sub-routine for step 1 of Algorithm 3
def func_step1(R, sorted_segments, lot_ids):

    '''
    Compute the Greedy solution for locations sorted in the descending order of
    the slopes of segments for Algorithm 3
    '''
    
    #Initialize parameters
    R_updated = R            #Keep track of number of remaining resources
    allocation_dict = {lot: 0 for lot in lot_ids}
    
    for segment in sorted_segments:
        resource_reqd = segment[-1]      #Compute resource required for segment

        if resource_reqd <= R_updated:
            R_updated -= resource_reqd                 #Update the amount of remaining resources
            allocation_dict[segment[0]] += resource_reqd    #Update the allocation of resources

        else:
            allocation_dict[segment[0]] += R_updated        #Update the allocation of resources
            
            break
            
    return allocation_dict

def func_step2(R, results2, lot_ids):
    
    '''
    Compute the allocation that maximizes welfare = W_R
    from spending at a single location
    '''
    
    allocation_dict = {lot: 0 for lot in lot_ids}
    
    welfare_single_loc = []
    for lot, data in results2:
        value = compute_location_welfare(R, data)
        welfare_single_loc.append((lot, value))
    
    max_value = max(value for _, value in welfare_single_loc)
    
    for j in welfare_single_loc:
        if j[1] == max_value:
            allocation_dict[j[0]] = min(R, 1)
    
    return max_value, allocation_dict