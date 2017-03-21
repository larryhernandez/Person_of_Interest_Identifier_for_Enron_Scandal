#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
#from sklearn.naive_bayes import GaussianNB

#############################################################################
###################     Define some useful functions        #################
#############################################################################

### To assist with visualizations

# This code for plotting the bar chart is borrowed and adapted directly from:
# http://matplotlib.org/examples/api/barchart_demo.html    


def print_underscore(n = 80):
    print "_" * n

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
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
    num_records, num_features = dataframe.shape
    num_poi = dataframe['poi'].sum()
    num_non_poi = df.poi.value_counts()[False]
    print "\tTotal number of records: " + str(num_records)
    print "\tTotal number of features: " + str(num_features)
    print "\tNumber of POIs: " + str(num_poi)
    print "\tNumber of non POIs: " + str(num_non_poi)
    return num_records, num_features, num_poi, num_non_poi

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
        df.insert(comp, 
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

##############################################################################
################### Task 1: Select what features you'll use. #################
##############################################################################

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
financial_feature_names = ['salary', 'deferral_payments', 'total_payments', 
                           'loan_advances', 'bonus', 
                           'restricted_stock_deferred', 'deferred_income', 
                           'total_stock_value', 'expenses', 
                           'exercised_stock_options', 'other', 
                           'long_term_incentive', 'restricted_stock', 
                           'director_fees']

email_feature_names = ['to_messages', 'from_poi_to_this_person', 
                       'from_messages', 'from_this_person_to_poi', 
                       'shared_receipt_with_poi']

# Select other control parameters for data processing, including: 
# feature scaling, feature selection & dimensionalit reduction, and model
# selection
required_financial_features = ['salary']
exclude_email_features = 0          # 0: utilize email features for modeling
                                    # 1: do not utilize email features

# Use StandardScaler
log_transform = 1
use_standard_scaler = 0

# Use MinMaxScaler for Feature Scaling
scale_email_features = 0            # irrelevant if exclude_email_features = 1
scale_financial_features = 0

# Feature Selection
create_new_email_features = 1       # irrelevant if exclude_email_features = 1
feature_select_emails = 1           # irrelevant if exclude_email_features = 1
n_email_features_to_select = 1      # irrelevant if exclude_email_features = 1
feature_select_finance = 1
n_finance_features_to_select = 3

# Define the target variable (useful for data cleansing)
target = 'poi'

# Control Variables for SelectKBest() and PCA()
#FEATURE_SELECTION_FN = [f_classif, chi2]
#N_FEATURES_OPTIONS = [2,3]
#N_PCA_COMPONENTS = [2,3,4]

##############################################################################
##############################################################################
#####            Load the dictionary containing the dataset              #####
##############################################################################
##############################################################################
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)    
##############################################################################
##############################################################################
#####                Exploratory Data Visualization & Analysis           #####
##############################################################################
##############################################################################

# Place raw data into pandas dataframe for easier manipulation and viewing
df = pd.DataFrame.from_dict(data_dict, orient = 'index')

# Number of records & available features in raw data set
print "Basic summary of raw data"
num_records_raw, num_features_raw, num_poi, num_non_poi = summarize_records(df)

# Remove the obvious outlier from dataframe
print "Removing major outlier"
outlier = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
df.drop(labels = outlier, inplace = True)

# Remove email addresses: these are just names that do not need to be included
df.drop(labels = 'email_address', axis = 1, inplace = True)

# Count the number of 'NaN' (strings) values occurring in each column
nan_counts = pd.DataFrame(data = None, index = [0], columns = df.columns, 
                             dtype = 'int')
nan_counts.poi = 0
for col_name in df.columns:
    if col_name != 'poi':
        num_nan = df[col_name].value_counts()['NaN']
        nan_counts.set_value(index = 0, col = col_name, value = num_nan)
print nan_counts

# Store the names of features which are mostly 'NaN'
columns_with_too_many_NaN = columns_with_more_than_x_NaN_vals(nan_counts, 
                                                          x = 90)

# Count the number of records missing relevant financial data. Since data 
# set is small and most Non-POIs lack financial information we should eliminate 
# these records from the classification task so as not to influence the results
num_records_missing_finance = 0
indices_missing_finance = []
for record in df.index.values:    
    comparison = (df.loc[record][required_financial_features] == 'NaN')
    if all(comparison):
        num_records_missing_finance += 1
        indices_missing_finance.append(record)

print str(num_records_missing_finance) + ' records are missing financial data.'

########
########                Basic Data Visualizations
########

# Replace 'NaN' string with np.nan to avoid errors when using plotting tools
df.replace(to_replace = 'NaN', value = np.nan, inplace = True)

### Allocation across classes: Bar chart of POIs, Non-POIs
num_poi = df.poi.value_counts()[True]
num_non_poi = df.poi.value_counts()[False]
print 'There are ' + str(num_poi) + ' POIs'
print 'There are ' + str(num_non_poi) + ' non-POIs'

fig, ax = plt.subplots()
rect1 = ax.bar(0.75,num_poi, width = 0.5, color = 'r')
rect2 = ax.bar(-0.25,num_non_poi, width = 0.5, color = 'g')
ax.set_ylabel('Count')
ax.set_title('Number of POIs and non-POIs')
ax.set_xticks([])
ax.legend((rect2[0], rect1[0]), ('Non-POIs','POIs'))
autolabel(rect1)
autolabel(rect2)
plt.show()

#### Summary Statistics and Histogram of Bonus
million = 10**6
hist_and_summary(df, 'bonus', scale = million, xlab = "Bonus [$MM]", 
                 ylab = "Counts", col = 'green')

#### Summary Statistics and Histogram of Salaries
hist_and_summary(df, 'salary', scale = million, xlab = "Salary [$ MM]", 
                 ylab = "Counts", col = 'red')

#### Summary Statistics and Histogram of Exercised Stock Options
hist_and_summary(df, 'exercised_stock_options', scale = million, 
                 xlab = 'Value of Exercised Stock [$ MM]', 
                 ylab = 'Counts', col = 'blue')

# Scatter Plot of Salary, Bonus
print "Scatter plot of Bonus vs Salary\n"
for record in df.index.values:
    if df.loc[record]['poi']:
        col = 'red'
    else:
        col = 'green'
    plt.plot(df.loc[record]['salary'] / million, df.loc[record]['bonus'] / million, 'o', color = col)
poi_patch = mpatches.Patch(color = 'red', label = 'POI')
non_poi_patch = mpatches.Patch(color = 'green', label = 'Non-POI')
plt.xlabel("Salary [$ MM]")
plt.ylabel("Bonus [$ MM]")
plt.legend(handles = [poi_patch, non_poi_patch])
axes = plt.gca()
axes.set_ylim(top = round(np.max(df.bonus/million)*1.10,0))
plt.show()
print_underscore()


# Scatter Plot of Emails To POI vs Emails from POI
for record in df.index.values:
    if df.loc[record]['poi']:
        col = 'red'
    else:
        col = 'blue'

    plt.plot(df.loc[record]['from_poi_to_this_person'], 
             df.loc[record]['from_this_person_to_poi'], '.',
             color = col)
poi_patch = mpatches.Patch(color = 'red', label = 'POI')
non_poi_patch = mpatches.Patch(color = 'blue', label = 'Non-POI')
plt.xlabel('From POI')
plt.ylabel('To POI')
plt.title('Number of Emails To/From POI')
plt.legend(handles = [poi_patch, non_poi_patch])
axes = plt.gca()
axes.set_ylim(top = round(np.max(df.from_this_person_to_poi)*1.10,0))
plt.show()
print_underscore()

##############################################################################         
####################   Task 2: Remove outliers                ################
##############################################################################
# Remove records missing financial information to avoid reducing effectiveness 
# of classifier
print "Removing records that are missing relevant financial data"
df.drop(labels = indices_missing_finance, inplace = True)
num_records, num_features, num_poi, num_non_poi  = summarize_records(df)

##############################################################################
### Task 3: Create new feature(s)
##############################################################################

if create_new_email_features:
    ##[1] Look at ratio of "from_poi_to_this_person" to "to_messages"
    df['fraction_email_from_poi'] = df.apply(fraction_email_from_poi, axis = 1)
    email_feature_names.append('fraction_email_from_poi')

    ##[2] Look at ratio of "from_poi_to_this_person" to "to_messages"
    df['fraction_email_to_poi'] = df.apply(fraction_email_to_poi, axis = 1)
    email_feature_names.append('fraction_email_to_poi')
    
    ##[3] Look at the ratio of "from_poi_to_this_person" to 
    # "from_this_person_to_POI" and compare the differences for POIs & non-POIs
    df['ratio_reciprocating_email_to_poi'] = \
                                df.apply(reciprocating_email_to_poi, axis = 1)
    email_feature_names.append('ratio_reciprocating_email_to_poi')

    # Visualize these new features
    hist_and_summary(df, 'ratio_reciprocating_email_to_poi')
    hist_and_summary(df, 'fraction_email_to_poi')
    hist_and_summary(df, 'fraction_email_from_poi')

# Re-arrange columns so that 'poi' is at the end
column_names = df.columns.tolist()
num_features = df.shape[1]
new_order = range(0,num_features)
poi_index = df.columns.get_loc(target)
new_order.pop(poi_index)
new_order.append(poi_index)
new_column_names = [column_names[ii] for ii in new_order]
df = df[new_column_names]

# Extract only the feature names
feature_names = df.columns
feature_names = feature_names.drop(target)
##############################################################################
### Feature Scaling and Selection
##############################################################################

# Remove columns that had more than 100 'NaN' values
df.drop(labels = columns_with_too_many_NaN, axis = 1, inplace = True)

# Update financial_feature_names to account for the fact that some of these
# columns were just removed from the dataframe
financial_feature_names = get_complement_list(financial_feature_names, 
                                              columns_with_too_many_NaN)

# Fill nan values with 0
df.fillna(value = 0, inplace = True)

# Logarithmic transformation of finance & email features
if log_transform:
    df[financial_feature_names]= np.log(1 + df[financial_feature_names])
    df[email_feature_names]= np.log(1 + df[email_feature_names])

if use_standard_scaler:
    std_scaler = StandardScaler()
    std_scaler.fit_transform(df[financial_feature_names])
    std_scaler.fit_transform(df[email_feature_names])

# Visualize some of the remaining features
print "\nHistograms after transformation \n"

hist_and_summary(df, 'bonus', scale = 1, xlab = "Log(Bonus)", 
                 ylab = "Counts", col = 'green')

#### Summary Statistics and Histogram of Salaries
hist_and_summary(df, 'salary', scale = 1, xlab = "Log(Salary)", 
                 ylab = "Counts", col = 'red')

#### Summary Statistics and Histogram of Exercised Stock Options
hist_and_summary(df, 'exercised_stock_options', scale = 1, 
                 xlab = 'Log(Value of Exercised Stock)', 
                 ylab = 'Counts', col = 'blue')

hist_and_summary(df, 'ratio_reciprocating_email_to_poi', col = 'red')
hist_and_summary(df, 'fraction_email_to_poi', col = 'green')
hist_and_summary(df, 'fraction_email_from_poi', col = 'orange')

# Feature Scaling
if scale_email_features or scale_financial_features:
    scaler = MinMaxScaler()

if exclude_email_features:
    df.drop(email_feature_names, axis = 1 , inplace = True)
else:
    if scale_email_features:
        print "Scaling Email Features with MinMaxScaler"
        X_email = scaler.fit_transform(df[email_feature_names])
        df[email_feature_names] = X_email
    else:
        X_email = df[email_feature_names]
    
if scale_financial_features:
    print "Scaling financial features with MinMaxScaler"
    X_fin = scaler.fit_transform(df[financial_feature_names])
    df[financial_feature_names] = X_fin
else:
    X_fin = df[financial_feature_names]

## Feature Selection
print "Performing Feature Selection"

## [1] Email Features: Use SelectKBest()
if (not exclude_email_features) and feature_select_emails:
    kbest = SelectKBest(chi2, k = n_email_features_to_select)
    y_email = df[target]
    X_email_kbest = kbest.fit_transform(X_email,y_email)
    indices_chosen_features = kbest.get_support(indices = True)
    selected_email_features = []
    print 'The best email features are: '
    print ""
    for x in indices_chosen_features:
        print email_feature_names[x]
        selected_email_features.append(email_feature_names[x])
    
    df[selected_email_features] = X_email_kbest
    email_features_to_drop = get_complement_list(email_feature_names, 
                                                 selected_email_features)
    df.drop(email_features_to_drop, axis = 1, inplace = True)

## [2] Financial features: Apply PCA
if feature_select_finance:
    pca = PCA(n_components = n_finance_features_to_select)
    pca.fit(X_fin)
    print pca.explained_variance_ratio_
    print "Explained variance is", round(sum(pca.explained_variance_ratio_),2)
    X_pca = pca.fit_transform(X_fin)
    df.drop(financial_feature_names, 
            axis = 1, inplace = True)           # Remove columns used for PCA
    insert_principal_components(df,X_pca)       # Insert principal components

# Update 'features_list' to include the new list of features
features_list = df.columns.tolist()
features_list.remove(target)
features_list.insert(0,target)

# Print the names of features that will be used for model development
print "Features to be used for modeling are:"
for index, item in enumerate(features_list[1:]):
    print "   [" + str(index+1) + "]", item

##############################################################################
################## Store to my_dataset for easy export below. ################
##############################################################################
# Print number of records & available features in revised data set
print "Basic summary of cleaned data set"
num_records_revised, num_features_revised, num_poi, num_non_poi = \
                                                        summarize_records(df)
my_dataset = df.to_dict(orient = 'index')

##############################################################################
### Extract features and labels from dataset for local testing
##############################################################################
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

##############################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
##############################################################################

# Control Variables for Cross-Validation
seed = 42
n_stratified_shuffle_splits = 30
test_size_sss = 0.10
grid_scoring = 'f1_micro'
NUM_TRIALS = 1

################  Pipeline and gridsearch for SVC
print "Creating Pipeline for Support Vector Classification"
pipe_svc = Pipeline([
                    ('classify', SVC(C = 1.0, 
                                     kernel = 'rbf',
                                     class_weight = 'balanced',                
                                     random_state = seed))
                ])

C_SVC = [2**-1,2**0,2**1,2**2,2**3]
GAMMA_SVC = [2**-2,1,2**2]
#C_SVC = [8]
#GAMMA_SVC = [0.25]
param_grid_svc =   [
        {
            'classify__C': C_SVC,
            'classify__gamma': GAMMA_SVC,
        }
                ]
############### Create Pipeline and GridSearch for Decision Tree
print "Creating Pipeline for Decision Tree"
pipe_dt = Pipeline([
                    ('classify', DecisionTreeClassifier(splitter = 'best',
                                                        presort = True,                                                        
                                                        random_state = seed))
                ])

CRITERION = ['gini','entropy']
SPLITTER = ['best', 'random']
MAX_FEATURES = ['sqrt','log2',1.0]
#MIN_SAMPLES_SPLIT = [2,4,6]
#MIN_SAMPLES_LEAF = [1,3]
#MAX_DEPTH = [None, 2,4,6]

param_grid_dt =   [
        {
            'classify__criterion': CRITERION,
#            'classify__splitter': SPLITTER,
#            'classify__max_features': MAX_FEATURES,
#            'classify__max_depth': MAX_DEPTH,
#            'classify__min_samples_split': MIN_SAMPLES_SPLIT,
#            'classify__min_samples_leaf': MIN_SAMPLES_LEAF
        }
                ]

################  Pipeline and gridsearchCV for Random Forest
print "Creating Pipeline for Random Forest"
pipe_rf = Pipeline([ 
                    ('classify', RandomForestClassifier(n_jobs = 1, 
                                                       random_state = seed))
                ])

N_ESTIMATORS_OPTIONS = [10]
RF_MAX_FEATURES = ['log2', 'sqrt']

param_grid_rf =   [
        {
            'classify__n_estimators': N_ESTIMATORS_OPTIONS,
            'classify__max_features': RF_MAX_FEATURES,
            'classify__criterion': CRITERION
        }
                  ]
############### Use Nested Cross-Validation to optimize hyperparameters and 
############### perform model evaluation during several trials
cv_scores_svc = np.zeros(NUM_TRIALS)
cv_scores_dt = np.zeros(NUM_TRIALS)
cv_scores_rf = np.zeros(NUM_TRIALS)

for trial in range(NUM_TRIALS):
    print "Trial # " + str(trial+1) + " of " + str(NUM_TRIALS)    
    inner_cv = StratifiedShuffleSplit(n_splits = n_stratified_shuffle_splits, 
                                      test_size = test_size_sss, 
                                      random_state = trial)
    
    outer_cv = StratifiedShuffleSplit(n_splits = n_stratified_shuffle_splits, 
                                      test_size = test_size_sss,
                                      random_state = trial + NUM_TRIALS)
    ####
    #### Random Forest Classifer
    ####
    grid_rf = GridSearchCV(pipe_rf, cv = inner_cv, scoring = grid_scoring,
                           param_grid = param_grid_rf)
    grid_rf.fit(features, labels)
    nested_score_rf = cross_val_score(grid_rf, X = features, y = labels, 
                                      cv = outer_cv)
    cv_scores_rf[trial] = nested_score_rf.mean()
   
    ####
    #### Support Vector Classifer
    ####
    grid_svc = GridSearchCV(pipe_svc, cv = inner_cv, scoring = grid_scoring,
                           param_grid = param_grid_svc)
    grid_svc.fit(features,labels)    
    nested_score_svc = cross_val_score(grid_svc, X=features, y = labels, 
                                      cv = outer_cv, scoring = grid_scoring)
    cv_scores_svc[trial] = nested_score_svc.mean()
    
    ####
    #### Decision Tree Classifer
    ####
    grid_dt = GridSearchCV(pipe_dt, cv = inner_cv, scoring = grid_scoring,
                           param_grid = param_grid_dt)
    grid_dt.fit(features,labels)
    nested_score_dt = cross_val_score(grid_dt, X=features, y = labels,
                                      cv = outer_cv, scoring = grid_scoring)
    cv_scores_dt[trial] = nested_score_dt.mean()

print "Estimated f1_micro scores"
print "\tRandom Forest:", cv_scores_rf
print "\tSVC:", cv_scores_svc
print "\tDecisionTree:", cv_scores_dt

plt.figure()
cv_trials = range(1,NUM_TRIALS+1)
svc_line, = plt.plot(cv_trials, cv_scores_svc, color = 'green')
dt_line, = plt.plot(cv_trials, cv_scores_dt, color = 'red')
rf_line, = plt.plot(cv_trials, cv_scores_rf, color = 'blue')
plt.legend([svc_line, dt_line, rf_line], 
           ['SVC', 'DecisionTree', 'RandomForest'])
plt.title(grid_scoring + ' scores')
plt.xlabel('Individual Trial #')
plt.ylabel("score", fontsize = '14')
axes = plt.gca()
axes.set_ylim(0,1)
plt.show()
##############################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

##############################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = grid_svc.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)