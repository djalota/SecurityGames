import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from helper_functions import *
from greedy_algorithm_helper import *
from func_helper_exponentialdistusers import *
import warnings
warnings.filterwarnings("ignore")

'''
NOTE: 
1. The distribution of user threshold probabilities at each location is exponentially distributed,
where the parameter lambda is calibrated using the (number of citations, allocation probability)
2. We take the probability of allocation to a location as that averaged across the 7 month horizon
3. Assume officer only needs to visit a location once every day, i.e., users remain at location
for entire day
4. We assume that citations are distributed at each location proportionally to the number of parking spaces
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
Populate the v_l and d_l values for all locations and permit types
'''
# Create a new DataFrame to count unique days in each month for each location
result_lot = df_enforcement.groupby(['LOC-CODE']).agg(
    number_of_unique_days=('DATE-TIME START', lambda x: x.dt.date.nunique())
).reset_index()

# Rename columns for clarity
result_lot.columns = ['Lot', 'Number of Unique Days']
result_lot['Allocation Probability'] = [i/210 for i in result_lot['Number of Unique Days']]

# Rename df_citations columns to match the month-year format in df_enforcement
df_citations.columns = ['Lot', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03']

# Melt df_citations
df_citations_long = pd.melt(df_citations, id_vars=['Lot'], var_name='Month', value_name='Total Citations')

# Convert Month in df_citations_long to period type for uniformity
df_citations_long['Month'] = pd.to_datetime(df_citations_long['Month']).dt.to_period('M')

greedy_allocation = []
greedy_welfare = []
existing_welfare = []
existing_allocation = []
max_values = []
random_welfare = []
other_greedy_welfare = []

multiplier_arr = [0.25, 0.5, 0.75, 1, 2, 3, 4, 5]

for multiplier in multiplier_arr:

    citations_per_lot = df_citations_long.groupby('Lot').agg({
        'Total Citations': 'sum'
    })
    citations_per_lot.reset_index(inplace=True)

    citations_per_lot['Citations per Day'] = [citations_per_lot['Total Citations'][i]/result_lot['Number of Unique Days'][i] for i in range(len(citations_per_lot['Total Citations']))]


    citations_per_lot['Citations per Day'] = citations_per_lot['Citations per Day']*multiplier

    merged_df = pd.merge(citations_per_lot, df_space, left_on='Lot', right_on='LOT')
    merged_df_frac = pd.merge(citations_per_lot, df_space, left_on='Lot', right_on='LOT')

    # Step 2: Calculate proportional citations for each permit type
    for permit in permit_types:
        merged_df[permit] = (merged_df[permit] / merged_df['TOTAL SPACES']) * merged_df['Citations per Day']
        
        for i in range(len(merged_df_frac[permit])):
            if merged_df_frac[permit][i]>0:
                merged_df_frac[permit][i] = (merged_df_frac['Citations per Day'][i] / merged_df_frac['TOTAL SPACES'][i])
            else:
                merged_df_frac[permit][i] = 0

    # Step 3: Create df_citations_by_permit DataFrame
    df_citations_by_permit = merged_df[['Lot'] + permit_types]

    # Step 3: Create df_citations_by_permit DataFrame
    df_citations_by_permit_frac = merged_df_frac[['Lot'] + permit_types]

    df_lot = pd.merge(df_citations_by_permit, result_lot, on=['Lot']) 
    df_lot_frac = pd.merge(df_citations_by_permit_frac, result_lot, on=['Lot']) 

    # Initialize a DataFrame to store lambda values
    df_lambda = df_lot_frac[['Lot']].copy()

    # Calculate lambda for each permit type and store in the new DataFrame
    for permit in permit_types:
        df_lambda[f'Lambda {permit}'] = -np.log(df_lot_frac[permit]) / df_lot_frac['Allocation Probability']

    df_lambda.replace([np.inf, -np.inf], 0, inplace=True)

    # Define the percentile ranges
    percentiles = np.arange(0, 1.1, 0.01)  # From 0 to 1 with step 0.1

    # Initialize DataFrame to store probabilities
    columns = [f'{permit_type} {int(p1 * 100)}-{int(p2 * 100)} percentile' 
            for permit_type in permit_types
            for p1, p2 in zip(percentiles[:-1], percentiles[1:])]
    df_percentiles = pd.DataFrame(index=df_lambda.index, columns=columns)

    # Calculate probabilities for each lambda and percentile range
    for index, row in df_lambda.iterrows():
        for col in df_lambda.columns:
            if 'Lambda' in col:
                if 'A PERMIT' in col:
                    permit_type = 'A PERMIT'
                elif 'C PERMIT' in col:
                    permit_type = 'C PERMIT'
                elif 'RESIDENT' in col:
                    permit_type = 'RESIDENT'
                elif 'RESIDENT/C' in col:
                    permit_type = 'RESIDENT/C'
                elif 'OTHER PERMIT' in col:
                    permit_type = 'OTHER PERMIT'
                else:
                    permit_type = 'VISITOR / TIMED SPACE'
                lambda_value = row[col]
                for p1, p2 in zip(percentiles[:-1], percentiles[1:]):
                    
                    column_name = f'{permit_type} {int(p1 * 100)}-{int(p2 * 100)} percentile'
                    # Calculate probability for the range using CDF
                    prob = -(1 - np.exp(-lambda_value * p1)) + (1 - np.exp(-lambda_value * p2))
                    # Assign probabilities to the dataframe
                    df_percentiles.at[index, column_name] = prob

    # Combine the results with the 'Lot' column from the original df_lambda for reference
    df_percentiles = pd.concat([df_lambda[['Lot']], df_percentiles], axis=1)

    # Create new columns for d_val for each percentile column
    for col in df_percentiles.columns:
        midpoint = get_midpoint(col)
        if midpoint is not None:
            df_percentiles[f'd val {col}'] = midpoint

    # Make sure the Lot columns match for a correct merge
    df_percentiles.rename(columns={'Lot': 'LOT'}, inplace=True)
    merged = pd.merge(df_percentiles, df_space, on='LOT')

    # Iterate over columns in df_percentiles to calculate new values
    for col in df_percentiles.columns:
        if 'percentile' in col and 'd_val' not in col:
            if 'A PERMIT' in col:
                permit_type = 'A PERMIT'
            elif 'C PERMIT' in col:
                permit_type = 'C PERMIT'
            elif 'RESIDENT' in col:
                permit_type = 'RESIDENT'
            elif 'RESIDENT/C' in col:
                permit_type = 'RESIDENT/C'
            elif 'OTHER PERMIT' in col:
                permit_type = 'OTHER PERMIT'
            else:
                permit_type = 'VISITOR / TIMED SPACE'
            new_col_name = f'Value of {col}'
            
            # Calculate new values
            merged[new_col_name] = merged[col] * merged[permit_type] * permit_dict[permit_type]

    '''
    Compute the list of all tuples that correspond to the upper and lower bound of the welfare function
    '''
    # List to store the result for each lot
    results = []
    results2 = []

    # Iterate over each row in DataFrame
    for _, row in merged.iterrows():
        
        #Generate pairs of value and threshold probabilities to each lot
        pairs = generate_types(row, percentiles)
        
        # Transform y values in pairs
        ub_pairs = compute_ub_points(pairs)
        lb_vals = compute_lb_points(pairs)

        # Append the result for this lot
        results.append((row['LOT'], ub_pairs))
        results2.append((row['LOT'], lb_vals))

    '''
    Compute the total number of available resources
    '''
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

    print(actual_welfare, final_obj_val)

    #Append the allocation and welfare values
    greedy_allocation.append(final_allocation)
    greedy_welfare.append(final_obj_val)
    existing_allocation.append(actual_alloc)
    existing_welfare.append(actual_welfare)
    max_values.append(sum([j_[1][-1][0] for j_ in results]))

    #Compute welfare of other strategies
    other_greedy_allocation = {'A': 0.14777777777777779, 'B': 0.14777777777777779, 'C': 0.14777777777777779, 'D': 0, 'E': 0.03333333333333333, 'F': 0.04, 'G': 0.042222222222222223, 'H': 0.14571428571428552, 'I': 0.14777777777777779}
    random_allocation = {'A': R_tot/9, 'B': R_tot/9, 'C': R_tot/9, 'D': R_tot/9, 'E': R_tot/9, 'F': R_tot/9, 'G': R_tot/9, 'H': R_tot/9, 'I': R_tot/9}
    other_greedy_wel = compute_total_welfare(other_greedy_allocation, results2)
    random_wel = compute_total_welfare(random_allocation, results2)
    other_greedy_welfare.append(other_greedy_wel)
    random_welfare.append(random_wel)

plot_welfare_num_strategic(multiplier_arr, existing_welfare, greedy_welfare, other_greedy_welfare, random_welfare)

max_val = sum([j_[1][-1][0] for j_ in results])

plot_welfare_frac_num_strategic(multiplier_arr, existing_welfare, greedy_welfare, other_greedy_welfare, random_welfare, max_val)

'''
Compare the Actual Allocation to the Greedy allocation
'''
print('Actual Allocation: %f', actual_alloc)
print('Greedy Allocation: %f', final_allocation)
print('Actual Welfare: %f', actual_welfare)
print('Greedy Welfare: %f', final_obj_val)
#print('Other Welfare %f', compute_total_welfare(other_greedy, results2))
print('Maximum Achievable Welfare: %f', max_val)

plot_func(actual_welfare, final_obj_val, max_val)