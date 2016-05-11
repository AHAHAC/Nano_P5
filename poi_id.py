#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import math
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### financial features: ['salary', 'deferral_payments', 'total_payments', 
###                      'loan_advances', 'bonus', 'restricted_stock_deferred',
###                      'deferred_income', 'total_stock_value', 'expenses',
###                      'exercised_stock_options', 'other', 'long_term_incentive',
###                      'restricted_stock', 'director_fees'] (all units are in US dollars)
### email features: ['to_messages', 'email_address', 'from_poi_to_this_person',
###                  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
###                 (units are generally number of emails messages; notable exception is email_address, which is a text string)
### POI label: [poi] (boolean, represented as integer)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'total_payments', 'exercised_stock_options', 
		'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'director_fees', 'to_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Data points: ", len(data_dict)
print data_dict[data_dict.keys()[0]]

### Task 2: Remove outliers

data_dict.pop( "TOTAL", 0 ) 

def num(s):
    try:
        ret = float(s)
        if math.isnan(ret):
        	return 0.0
        else:
        	return ret
    except ValueError:
        return 0.0

data_sorted = sorted( zip(data_dict.keys(), 
			[ data_dict[a]["salary"] for a in data_dict], 
			[ data_dict[b]["bonus"] for b in data_dict],
			[ data_dict[c]["exercised_stock_options"] for c in data_dict]),
		key=lambda x:( num(x[2]), num(x[1]), num(x[3]), x[0]), reverse = True)[0:int(round(len(data_dict)*0.1))]

for x in data_sorted:
	data_dict.pop(x[0],0)
	print "Popped: ", x[0]


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for x in my_dataset:
	my_dataset[x]['salary'] = num(my_dataset[x]['salary']) + num(my_dataset[x]['director_fees'])
	print 'New salary:', my_dataset[x]['salary']
	if num(my_dataset[x]['total_payments']) == 0.0 or math.isnan(num(my_dataset[x]['total_payments'])):
		my_dataset[x]['ratio_sal_totalp'] = 0.0
	else:
		if math.isnan(num(my_dataset[x]['salary'])):
			my_dataset[x]['ratio_sal_totalp'] = 0.0
		else:
			my_dataset[x]['ratio_sal_totalp'] = num(my_dataset[x]['salary'])/num(my_dataset[x]['total_payments'])
			#my_dataset[x]['ratio_sal_totalp'] = num(my_dataset[x]['salary'])/num(my_dataset[x]['exercised_stock_options'])
			if my_dataset[x]['poi']:
				print 'POI:', num(my_dataset[x]['salary'])/num(my_dataset[x]['total_payments'])
				print 'salary:', num(my_dataset[x]['salary'])
				print 'total paymnt:', num(my_dataset[x]['total_payments'])
			else:
				print 'NON-POI:', num(my_dataset[x]['salary'])/num(my_dataset[x]['total_payments'])		

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print data

count_poi = 0
count_all = len(data)

for point in data:
    salary = point[1]
    bonus = point[2]
    tpayments = point[3]
    exoptions = point[4]
    sal_total = point[7]
    director_fee = point[8]
    if point[0] != 0.0 :
    	#POI
    	count_poi = count_poi + 1
    	#matplotlib.pyplot.scatter( salary, tpayments, c='blue' )
    	#matplotlib.pyplot.scatter( salary, bonus, c='yellow' )
    	#matplotlib.pyplot.scatter( salary, sal_total, c='red' )
    	matplotlib.pyplot.scatter( salary, director_fee, c='yellow' )
    #else:
        # Non POI
    	#matplotlib.pyplot.scatter( salary, tpayments, c='white' )
    	#matplotlib.pyplot.scatter( salary, bonus, c='red' )
    	#matplotlib.pyplot.scatter( salary, sal_total, c='black' )
    	#matplotlib.pyplot.scatter( salary, director_fee, c='black' )
    	
    #matplotlib.pyplot.scatter( salary, exoptions, c='green' )
    
    
#matplotlib.pyplot.xlabel("salary")
#matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

print count_poi

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)