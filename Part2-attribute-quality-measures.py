# Description: This script contains functions to calculate 3 different types of 
# including Information Gain, Gini Impurity, and Chi-square statistic.
# 
# 
# Created on: 9th november 2024
# --------------------------------------------------
# Libraries required: 
#   - numpy
#   - math 
#
# Input: a 2x2 numpy array as input.
#
# Output: Each function returns a float value
# --------------------------------------------------


import numpy as np
from math import log



def get_gain(contingency_table):
    """
    Calculates the Information Gain of an attribute by comparing the entropy of the dataset 
    before and after splitting based on the specific attribute.


    Parameters:
    contingency_table : 2x2 numpy array [[a, b], [c, d]], 
    
    where:

    a = count(attribute=yes, diagnosis=positive),
    b = count(attribute=yes, diagnosis=negative),
    c = count(attribute=no, diagnosis=positive),
    d = count(attribute=no, diagnosis=negative)
    
    Returns:
    a float of the Information Gain of the contingency table
    """
    #get p values
    total = np.sum(contingency_table)
    
    p_positive = (contingency_table[0, 0] + contingency_table[1, 0]) / total 
    p_negative = (contingency_table[0, 1] + contingency_table[1, 1]) / total

    # calc for entropy root
    H_root = -(p_positive * np.log2(p_positive) + p_negative * np.log2(p_negative))

    
    H_pos = 0
    H_neg = 0

    # calc entropy for positive and negative
    p_yes_pos = contingency_table[0, 0] / contingency_table[0].sum() 
    p_yes_neg = contingency_table[0, 1] / contingency_table[0].sum()
    H_pos = -(p_yes_pos * np.log2(p_yes_pos) + p_yes_neg * np.log2(p_yes_neg)) if p_yes_pos > 0 and p_yes_neg > 0 else 0
    
    p_no_pos = contingency_table[1, 0] / contingency_table[1].sum()
    p_no_neg = contingency_table[1, 1] / contingency_table[1].sum()
    H_neg = -(p_no_pos * np.log2(p_no_pos) + p_no_neg * np.log2(p_no_neg)) if p_no_pos > 0 and p_no_neg > 0 else 0
    
    # calc entropy after split
    entropy_after = (contingency_table[0].sum() / total) * H_pos + (contingency_table[1].sum() / total) * H_neg
    
    # compare root with after to get information gain
    gain = H_root - entropy_after
    return gain



 
def get_gini(contingency_table):
    """
    Calculates the Gini impurity 

    

    Parameters:
    contingency_table : 2x2 numpy array [[a, b], [c, d]], 
    
    where:

    a = count(attribute=yes, diagnosis=positive),
    b = count(attribute=yes, diagnosis=negative),
    c = count(attribute=no, diagnosis=positive),
    d = count(attribute=no, diagnosis=negative)
    


    Returns:
    a float of  the Gini impurity of an attribute
    """
    # Get p values from table
    total = np.sum(contingency_table)
    
    p_yes = contingency_table[0].sum() / total
    p_no = contingency_table[1].sum() / total

    #get impurity of each split
    gini_yes = 1 - ((contingency_table[0, 0] / contingency_table[0].sum()) ** 2 + (contingency_table[0, 1] / contingency_table[0].sum()) ** 2) if contingency_table[0].sum() > 0 else 0
    gini_no = 1 - ((contingency_table[1, 0] / contingency_table[1].sum()) ** 2 + (contingency_table[1, 1] / contingency_table[1].sum()) ** 2) if contingency_table[1].sum() > 0 else 0

    #Calculate giniimpurity
    gini = p_yes * gini_yes + p_no * gini_no
    return gini

    



def get_chi(contingency_table):
    """
    Calculates the Chi-square statistic 

    Parameters:
    contingency_table : 2x2 numpy array [[a, b], [c, d]], 
    
    where:

    a = count(attribute=yes, diagnosis=positive),
    b = count(attribute=yes, diagnosis=negative),
    c = count(attribute=no, diagnosis=positive),
    d = count(attribute=no, diagnosis=negative)
    
    Returns:
    a float Chi-square statistic of a given attribute
    """
     
    observed = contingency_table

    # finds the amount of elements in each row and column
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total = observed.sum()
    
    #initialize statistic
    x2 = 0.0

    # Loop through the rows and columns of the table
    for i in range(2):
        for j in range(2):
            
            #calculate count for each
            expected = (row_sums[i] * col_sums[j]) / total
            
            #Add the Chi-square contribution for this cell 
            if expected > 0:
                x2 += (observed[i, j] - expected) ** 2 / expected

    return x2






#contingency_table_test =  np.array([[10, 20], [30, 40]])
#contingency_table_test =  np.array([[0, 0], [4, 8]]) zero error checking works fine
#contingency_table_test =  np.array([[10, 10], [10,10]]) works correctly with all values the same
#contingency_table_test =  np.array([0, 0], ) doesnt execute when input a invalid array



"""""
print("Test case1")
print("Information Gain:", get_gain(contingency_table_test))
print("Gini Impurity:", get_gini(contingency_table_test))
print("Chi-square:", get_chi(contingency_table_test))
print("\n")
"""


contingency_table_headache = np.array([[3, 0], [2, 3]])
contingency_table_spots = np.array([[4, 1], [1, 2]])
contingency_table_stiff_neck = np.array([[4, 1], [1, 2]])




print("Headache Attribute")
print("Information Gain:", get_gain(contingency_table_headache))
print("Gini Impurity:", get_gini(contingency_table_headache))
print("Chi-square:", get_chi(contingency_table_headache))
print("\n")

print("Spots Attribute")
print("Information Gain:", get_gain(contingency_table_spots))
print("Gini Impurity:", get_gini(contingency_table_spots))
print("Chi-square:", get_chi(contingency_table_spots))
print("\n")

print("Stiff Neck Attribute")
print("Information Gain:", get_gain(contingency_table_stiff_neck))
print("Gini Impurity:", get_gini(contingency_table_stiff_neck))
print("Chi-square:", get_chi(contingency_table_stiff_neck))