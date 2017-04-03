# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:23:13 2017

@author: Larry
"""
import numpy as np
import matplotlib.pyplot as plt

#############################################################################
###################     Define some useful functions        #################
#############################################################################

### To assist with visualizations

def print_underscore(n = 80):
    print "_" * n


def autolabel(rects,fig_axes):
    """
    Attach a text label above each bar displaying its height
    This code for plotting the bar chart is borrowed and adapted directly from:
    http://matplotlib.org/examples/api/barchart_demo.html    
    """
    for rect in rects:
        height = rect.get_height()
        fig_axes.text(rect.get_x() + rect.get_width()/2., height,
                '%d' % int(height),
                ha='center', va='bottom')


def hist_and_summary(df, column, scale = 1, xlab ='', ylab = '', col = 'blue'):
    """
    Given a pandas dataframe and column names, prints summary statistics for 
    that column and produces a histogram.
    """
    # Summary Statistics
    print df[column].describe()
    
    # Histogram
    plt.hist(df[column][np.isfinite(df[column])] / scale, 
             bins = 50, color = col)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()


def summarize_records(dataframe):
    '''
    Given a dataframe (of enron employee data) prints the number of POIs, 
    non-POIs, number of columns, and total number of records.
    '''
    num_records, num_columns = dataframe.shape
    num_poi = dataframe['poi'].sum()
    num_non_poi = dataframe.poi.value_counts()[False]
    print "\tTotal number of records: " + str(num_records)
    print "\tTotal number of columns: " + str(num_columns)
    print "\tNumber of POIs: " + str(num_poi)
    print "\tNumber of non POIs: " + str(num_non_poi)
    return num_records, num_columns, num_poi, num_non_poi

##### Functions to assist with eliminating or re-arranging columns

def columns_with_more_than_x_NaN_vals(dataframe, x = 100):
    identified_columns = []
    for name in dataframe.columns.tolist():
        if dataframe.loc[0][name] >= x:
            identified_columns.append(name)
    return identified_columns

##### Functions to assist with Feature Creation

# This function was borrowed from the Udacity video lecture entitled 
# "Quiz: Visualizing Your New Feature"
def computeFraction( poi_messages, all_messages ):
    '''
    Given a number messages to/from POI (numerator) and number of all messages 
    to/from a person (denominator), return the fraction of messages to/from 
    that person that are from/to a POI
    '''
    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if (poi_messages != "NaN" or poi_messages != np.nan) and (all_messages != "NaN" or all_messages!= np.nan):
        if (np.isnan(poi_messages) or poi_messages == 0.0) or (np.isnan(all_messages) or all_messages == 0.):
            fraction = 0.            
        else:
            fraction = poi_messages / float(all_messages)  
    return fraction


def fraction_email_from_poi(row):
    '''
    Calculates the fraction of emails that a given person (i.e. row) receives
    from POIs
    '''
    return computeFraction(row['from_poi_to_this_person'], row['to_messages'])


def fraction_email_to_poi(row):
    '''
    Calculates the fraction of emails that a given person (i.e. row) sends
    to POIs
    '''
    return computeFraction(row['from_this_person_to_poi'], 
                           row['from_messages'])


def reciprocating_email_to_poi(row):
    '''
    Calculates the ratio where the numerator is "number of emails that a person 
    sends to POIs" and the denominator is "number of emails that a person 
    receives from POIs"
    '''
    return computeFraction(row['from_this_person_to_poi'], 
                           row['from_poi_to_this_person'])


#### To assist with Feature Selection
def insert_principal_components(dataframe, C):
    '''
    Given a dataframe containing columns that were used for PCA, and the 
    resultant principal components stored in 'C', this function inserts each 
    principal component as a column in dataframe.
    '''
    [num_samples, num_components] = C.shape
    for comp in range(0,num_components):
        col_name = 'PC_' + str(comp+1)
        dataframe.insert(comp, 
                         column = col_name, 
                         value = C[:,comp], 
                         allow_duplicates = False)


def get_complement_list(first, second):
    '''
    Given two lists, denoted first and second, returns a list of items which
    are contained in first but not contained in second.
    '''
    return list(set(first) - set(second))

###
### End of useful functions section
###