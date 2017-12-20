#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot 
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL",0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below



person_outlier = ""

salary = 0

for key in data_dict:
    if data_dict[key]["salary"] >  salary and data_dict[key]["salary"] != "NaN":
        salary = data_dict[key]["salary"]
        person_outlier = key

print person_outlier
print salary



for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

person = ""

for key in data_dict:
    if  data_dict[key]["salary"] != "NaN" and data_dict[key]["salary"] > 1e+6 and data_dict[key]["bonus"] > 5e+6:
        print key

##for key in data_dict:
##    if data_dict[key]["bonus"] > 1e+6 and data_dict[key]["bonus"] != "NaN":
##        print key




