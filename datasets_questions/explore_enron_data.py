#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

##print len(enron_data)
##
##print len(enron_data[enron_data.keys()[0]])
##
### or print len(enron_data["BUY RICHARD B"].keys())
##
count = 0
for person in enron_data.values():
    if person['poi'] is True:
        count = count + 1
##
print "Person of Interests:", count

## or print len(dict((key, value) for key, value in enron_data.items() if value["poi"] == True))

##poi_reader = open('../final_project/poi_names.txt', 'r')
##poi_reader.readline() # skip url
##poi_reader.readline() # skip blank line
##
##poi_count = 0
##for poi in poi_reader:
##	poi_count = poi_count + 1
##
##print poi_count
##
##print enron_data["PRENTICE JAMES"]["total_stock_value"]
##
##print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
##
##print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
##
##person_paid_most = ''
##
##payment = 0
##
##for key in ('SKILLING JEFFREY K','FASTOW ANDREW S','LAY KENNETH L'):
##    if enron_data[key]['total_payments'] > payment:
##        payment = enron_data[key]['total_payments']
##        person_paid_most = key
##print person_paid_most, payment
##
##print(enron_data['SKILLING JEFFREY K'])
##
##ccc = 0
##for value in enron_data.values():
##    if value["salary"] != "NaN":
##        ccc = ccc +1
##
##print ccc
##
##email_address = 0
##
##for key in enron_data.values():
##    if key["email_address"] != "NaN":
##       email_address = email_address +1
##print email_address
##
NaN_count = 0
for key in enron_data.values():
    if key["total_payments"] == "NaN":
        NaN_count = NaN_count + 1
##print "the number of NaN:", NaN_count
##
##print "the percentage of NaN:", NaN_count/float(len(enron_data))*100
##

Poi = 0
for key, value in enron_data.items():
    if value["poi"] is True and value["total_payments"] == "NaN":
        Poi = Poi +1
print Poi
print Poi/float(len(enron_data))*100

print NaN_count+10
print len(enron_data)+10

print count+10
print Poi+10

##print (Poi+10)/float(len(enron_data)+10)*100


