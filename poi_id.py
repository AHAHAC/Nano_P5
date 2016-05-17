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
features_list = ['poi', 'salary', 'bonus', 'total_payments', 'exercised_stock_options', 'director_fees'] # You will need to use more features
		#'shared_receipt_with_poi',
		#, 'to_messages',
		#'from_poi_to_this_person', 'from_this_person_to_poi'

print features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Total data points: ", len(data_dict)

### Task 2: Remove outliers
#Remove "TOTAL" line
data_dict.pop( "TOTAL", 0 ) 
#define a function to convert string to float. Used for removing outliers.
def num(s):
    try:
        ret = float(s)
        if math.isnan(ret):
        	return 0.0
        else:
        	return ret
    except ValueError:
        return 0.0
#Let's sort data by bonuses, salary, exercised stock options. And get 10% of points.
data_sorted = sorted( zip(data_dict.keys(), 
			[ data_dict[a]["salary"] for a in data_dict], 
			[ data_dict[b]["bonus"] for b in data_dict],
			[ data_dict[c]["exercised_stock_options"] for c in data_dict]),
		key=lambda x:( num(x[2]), num(x[1]), num(x[3]), x[0]), reverse = True)[0:int(round(len(data_dict)*0.1))]
#Remove 10% of data points with the highest bonuses
for x in data_sorted:
	data_dict.pop(x[0],0)
	print "Popped: ", x[0]

#Print 5 datapoint
i = 0
for x in data_dict:
	print x
	print data_dict[x]
	print "--------------"
	i = i + 1
	if i >= 5:
		break

print "Data points after outliers removal:", len(data_dict)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
poi_count = 0
#Let create a new feature: ratio of salary and total payments
for x in my_dataset:
	if math.isnan(num(my_dataset[x]['total_payments'])):
		my_dataset[x]['ratio_sal_totalp'] = 0.0
	else:
		if num(my_dataset[x]['total_payments']) == 0.0:
			my_dataset[x]['ratio_sal_totalp'] = 0.0
		else:
			my_dataset[x]['ratio_sal_totalp'] = num(my_dataset[x]['salary'])/num(my_dataset[x]['total_payments'])
	if math.isnan(num(my_dataset[x]['to_messages'])) | (num(my_dataset[x]['to_messages']) == 0.0):
		my_dataset[x]['to2from_poi_to_this_person'] = 0.0
	else:
		my_dataset[x]['to2from_poi_to_this_person'] = num(my_dataset[x]['from_poi_to_this_person'])/num(my_dataset[x]['to_messages'])
	if math.isnan(num(my_dataset[x]['from_messages'])) | (num(my_dataset[x]['from_messages']) == 0.0):
		my_dataset[x]['from2from_this_person_to_poi'] = 0.0
	else:
		my_dataset[x]['from2from_this_person_to_poi'] = num(my_dataset[x]['from_this_person_to_poi'])/num(my_dataset[x]['from_messages'])

features_list.append("ratio_sal_totalp")
features_list.append("to2from_poi_to_this_person")
features_list.append("from2from_this_person_to_poi")

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = False) #17/05/2016 changed sort_keys from True to False
labels, features = targetFeatureSplit(data)

count_poi = 0
count_nonpoi = 0
count_all = len(data)

print "Count all after feature split:", count_all
print "-----------3-------------"
print "0:",data[0]
print "1:",data[1]
print "2:",data[2]
print "3:",data[3]
print "-----------4-------------"

for point in data:
    salary = point[1]
    bonus = point[2]
    tpayments = point[3]
    exoptions = point[4]
    sal_total = point[7]
    #director_fee = point[8]
    #print point[0]
    if float(point[0]) == 0.0 :
    	# Non POI
	#matplotlib.pyplot.scatter( salary, tpayments, c='white' )
	matplotlib.pyplot.scatter( salary, bonus, c='red' )
	#matplotlib.pyplot.scatter( salary, sal_total, c='black' )
	#print salary,':', sal_total
	count_nonpoi = count_nonpoi + 1
    	#matplotlib.pyplot.scatter( salary, director_fee, c='black' )
    else:
    	#POI
	count_poi = count_poi + 1
	#matplotlib.pyplot.scatter( salary, tpayments, c='blue' )
	matplotlib.pyplot.scatter( salary, bonus, c='yellow' )
	#matplotlib.pyplot.scatter( salary, sal_total, c='red' )
    	#matplotlib.pyplot.scatter( salary, director_fee, c='yellow' )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()

print "POIs: ", count_poi
print "NON-POIs: ", count_nonpoi

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf.fit(features, labels)



#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values=0, strategy='mean', axis=0)
#imp.fit(features)
#features = imp.transform(features)

#from sklearn.svm import SVC
#clf = SVC(verbose=True)
#clf.fit(features, labels) 

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(features, labels)


from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

estimators = [('reduce_dim', PCA(n_components=4)), ('decisontree', DecisionTreeClassifier(random_state=0, max_depth=3))]
clf = Pipeline(estimators)

clf.fit(features, labels)
print features[0]
print features[1]
print features[2]
#pca = PCA(n_components=3)
#pca = PCA()


#print features_list
#print features[10]
#print features[30]
#features = pca.fit_transform(features)
#print features[10]
#print features[30]
pca = clf.named_steps['reduce_dim']
print("PCA Explained: ", pca.explained_variance_ratio_) 
print pca.components_ 


#clf = DecisionTreeClassifier(random_state=0, max_depth=3)
#clf.fit(features, labels)

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