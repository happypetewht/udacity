#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
from datetime import date
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features =  ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_labels = ['poi']
features_list = poi_labels + financial_features + email_features
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


df = pd.DataFrame.from_dict(data_dict,orient='index')
df = df.replace('NaN',np.nan)





### Task 2: Remove outliers

#missing values

df[financial_features] = df[financial_features].fillna(0)
df[email_features] = df[email_features].fillna(df[email_features].median())



#explore dataset. this is basic exploration. Need to know the number of features and lists
# use dataframe to explore dataset

df = df.drop('TOTAL')

df = df[df.from_messages < 6000]
df = df[df.to_messages <10000]


##data_points = len(data_dict)
##poi_count = 0
##non_poi_count = 0
##
##print "Data Point:", data_points
##for k in data_dict:
##    if data_dict[k]["poi"] == True:
##        poi_count = poi_count +1
##print "poi number:", poi_count
##
##for k in data_dict:
##    if data_dict[k]["poi"] == False:
##        non_poi_count = non_poi_count +1
##
##all_features = data_dict[data_dict.keys()[0]].keys()
##print("all_features:",all_features)
##print "non poi number:", non_poi_count
##print "total features:", len(data_dict[data_dict.keys()[0]])
##print "financial features:", len(financial_features)
##print "email features:", len(email_features)
##
##print("There are %i features for each person in the dataset, and %i features \
##are used" %(len(all_features), len(features_list)))
##
# check the missing value

##total_missing_count = 0
##
##missing_value = {}
##
##for feature in all_features:
##    missing_value[feature] = 0
##
##
##
##for person in data_dict:
##    for feature in data_dict[person]:        
##        if   data_dict[person][feature] == "NaN":
##            total_missing_count = total_missing_count +1
##            missing_value[feature] += 1
##
##print('the total missing number of features:',total_missing_count)
##print("the number of missing values of each feature:")
##for feature in  missing_value:
##    print("%s: %i" %(feature, missing_value[feature]))

##missing_values = {}
##for feature in all_features:
##    missing_values[feature] = 0
##for person in data_dict:
##    for feature in data_dict[person]:
##        if data_dict[person][feature] == "NaN":
##            missing_values[feature] += 1
##print("The number of missing values for each feature: ")
##for feature in missing_values:
##    print("%s: %i" %(feature, missing_values[feature]))


### Task 3: Create new feature(s)

df['fraction_of_messages_to_poi'] = df.from_this_person_to_poi / df.from_messages
df['fraction_of_messages_from_poi'] = df.from_poi_to_this_person / df.to_messages


### Store to my_dataset for easy export below.
my_dataset = df.to_dict('index')

print my_dataset.head()

features_list = [u'poi', u'salary', u'to_messages', u'deferral_payments', u'total_payments',
       u'exercised_stock_options', u'bonus', u'restricted_stock',
       u'shared_receipt_with_poi', u'expenses', u'from_messages', u'other',
       u'long_term_incentive', u'fraction_of_messages_to_poi',
       u'fraction_of_messages_from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

selection = SelectKBest(k=7)
selection.fit(features,labels)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier



#clf = GaussianNB()
#clf = DecisionTreeClassifier()
#clf = LogisticRegression()
#clf = AdaBoostClassifier()
clf= RandomForestClassifier()
##clf = LGBMClassifier(params=params,max_bin=200)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.metrics import accuracy_score,precision_score, recall_score

clf.fit(features_train,labels_train)

# Naive Bayes classifier no changes for K
# k = [1, 2, 3,  4, 5,  6, 7, 8, 9]
# accuracy =[0.81820,]
# precision = [0.24091,]
# recall = [0.16900,]

# Decision Tree

#      k     :    1     2     3      4      5      6      7     8        9    10      11   
#('accuracy:', 0.821, 0.822, 0.823, 0.822, 0.824, 0.822, 0.824, 0.822, 0.821, 0.821, 0.814
#('precision:',0.318, 0.320, 0.326, 0.324, 0.329, 0.323, 0.330  0.324, 0.321, 0.321  , 0.333
#('recall:',   0.300, 0.300, 0.311, 0.308, 0.308, 0.309, 0.312  0.304, 0.305, 0.306, 0.333

# logistic
# accuract = 0.80707, precision = 0.19550, Recall: 0.14350


# AdaBoostClassifier
# all default: Accuracy: 0.85640	Precision: 0.44974	Recall: 0.34450
# max_depth:1, n_estimator =100, Accuracy: 0.84900	Precision: 0.40830  Recall: 0.29500
# max_depth 2  n_estimator =50  Accuracy: 0.83740	Precision: 0.32703	Recall: 0.20750
# max_depth 2  n_estimator =200  Accuracy: 0.83533	Precision: 0.30417	Recall: 0.18250


# Random Forest
# default Accuracy: 0.86033	Precision: 0.42301	Recall: 0.13050

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
