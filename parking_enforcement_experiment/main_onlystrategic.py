import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from helper_functions import *
from greedy_algorithm_helper import *
from func_helper_onlystrategic import *

'''
NOTE: 
1. All users assumed to be strategic behaving rationally as specified by theory
2. We take the probability of allocation to a location as that averaged across the 7 month horizon
3. Assume officer only needs to visit a location once every day, i.e., users remain at location
for entire day
'''

'''
Read Data
'''
xls = pd.ExcelFile('ParkEnforcementModelData.xlsx')
df_enforcement = pd.read_excel(xls, 'Enforcement Time')
df_space = pd.read_excel(xls, 'Space Allocation')
df_citations = pd.read_excel(xls, 'Citations (7 months)')

df_space.fillna(0, inplace = True)

'''
Store data for each fraction of strategic users
'''
greedy_allocation = []
greedy_welfare = []
existing_welfare = []
existing_allocation = []
max_values = []
random_welfare = []

strategic_arr = np.linspace(0, 1, 11)

for frac_strategic in strategic_arr:

    if frac_strategic == 0:

        greedy_welfare.append(0)
        existing_welfare.append(0)
        random_welfare.append(0)

    else:
        '''
        Populate the v_l and d_l values for all locations and permit types
        '''
        #Calibrate the benefits accrued by each user type for engaging in fraud
        df_space = compute_value_onlystrategic(df_space, permit_dict, frac_strategic)

        #Calibrate the threshold probabilities for each permit type
        df_space = compute_d_val_onlystrategic(df_space, permit_dict, threshold_prob)

        '''
        Compute the list of all tuples that correspond to the upper and lower bound of the welfare function
        '''
        # List to store the result for each lot
        results = []
        results2 = []

        # Iterate over each row in DataFrame
        for _, row in df_space.iterrows():
            
            #Generate pairs of value and threshold probabilities to each lot
            pairs = generate_types(row)
            
            # Transform y values in pairs
            ub_pairs = compute_ub_points(pairs)
            lb_vals = compute_lb_points(pairs)

            # Append the result for this lot
            results.append((row['LOT'], ub_pairs))
            results2.append((row['LOT'], lb_vals))

        '''
        Compute the total number of available resources
        '''
        # Create a new DataFrame to count unique days in each month for each location
        result_lot = df_enforcement.groupby(['LOC-CODE']).agg(
            number_of_unique_days=('DATE-TIME START', lambda x: x.dt.date.nunique())
        ).reset_index()

        # Rename columns for clarity
        result_lot.columns = ['Lot', 'Number of Unique Days']
        result_lot['Allocation Probability'] = [i/210 for i in result_lot['Number of Unique Days']]

        #Amt of resources available
        R_tot = sum(result_lot['Allocation Probability'])

        '''
        Algorithm Implementation Block
        '''
        #Step 1: Compute Solution and welfare for step 1

        # Step 1a: Compute the convex hull of the points and sort the segments
        hulls = {}
        for lot, pairs in results:
            # Adding the origin point
            pairs = pairs
            # Compute the upper convex hull
            hulls[lot] = upper_convex_hull(pairs)

        sorted_segments = list_of_slopes(hulls)

        #Step 1b: Compute the optimal allocation corresponding to step 1
        lot_ids = df_space['LOT'].values
        allocation_step1 = func_step1(R_tot, sorted_segments, lot_ids)
        obj_val_step1 = compute_total_welfare(allocation_step1, results2)

        #Step 2: Compute solution and welfare for step 2
        obj_val_step2, allocation_step2 = func_step2(R_tot, results2, lot_ids)

        #Step 3: Compute allocation for best of the two steps
        if obj_val_step1 >= obj_val_step2:
            final_obj_val = obj_val_step1
            final_allocation = allocation_step1
        else:
            final_obj_val = obj_val_step2
            final_allocation = allocation_step2

        '''
        Computation of Welfare for Current Policy Used in Practice
        '''
        #Compute the actual allocation and its corresponding welfare
        actual_allocation = {lot: 0 for lot in lot_ids}
        actual_welfare_val = 0
        for lot, data in results2:
            if lot in result_lot['Lot'].values:
                amt_allocated = result_lot[result_lot['Lot'] == lot]['Allocation Probability'].values[0]
                welfare = compute_location_welfare(amt_allocated, data)
                actual_allocation[lot] = amt_allocated
                actual_welfare_val += welfare
            else:
                actual_welfare_val += 0

        #Append the allocation and welfare for the months
        actual_alloc = actual_allocation
        actual_welfare = actual_welfare_val

        #Append the allocation and welfare values
        greedy_allocation.append(final_allocation)
        greedy_welfare.append(final_obj_val)
        existing_allocation.append(actual_alloc)
        existing_welfare.append(actual_welfare)
        max_values.append(sum([j_[1][-1][0] for j_ in results]))

        #Add results for random allocation of resources
        random_allocation = {'A': R_tot/9, 'B': R_tot/9, 'C': R_tot/9, 'D': R_tot/9, 'E': R_tot/9, 'F': R_tot/9, 'G': R_tot/9, 'H': R_tot/9, 'I': R_tot/9}
        random_wel = compute_total_welfare(random_allocation, results2)
        random_welfare.append(random_wel)

plot_welfare_num_strategic(strategic_arr, existing_welfare, greedy_welfare, random_welfare)

max_val = sum([j_[1][-1][0] for j_ in results])

plot_welfare_frac_num_strategic(strategic_arr, existing_welfare, greedy_welfare, random_welfare, max_val)

print(existing_welfare[5], greedy_welfare[5], max_val)

'''
Compare the Actual Allocation to the Greedy allocation
'''
print('Actual Allocation: %f', actual_alloc)
print('Greedy Allocation: %f', final_allocation)
print('Random Allocation: %f', random_allocation)
print('Actual Welfare: %f', actual_welfare)
print('Greedy Welfare: %f', final_obj_val)
print('Maximum Achievable Welfare: %f', max_val)

plot_func(actual_welfare, final_obj_val, max_val)