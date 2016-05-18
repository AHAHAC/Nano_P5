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
print "Data points after outliers removal:", len(data_dict)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#define a function to convert string to float.
def num(s):
    try:
        ret = float(s)
        if math.isnan(ret):
        	return 0.0
        else:
        	return ret
    except ValueError:
        return 0.0

my_dataset = data_dict
poi_count = 0
#Let create a new feature: ratio of salary and total payments
#for x in my_dataset:
#	if math.isnan(num(my_dataset[x]['total_payments'])):
#		my_dataset[x]['ratio_sal_totalp'] = 0.0
#	else:
#		if num(my_dataset[x]['total_payments']) == 0.0:
#			my_dataset[x]['ratio_sal_totalp'] = 0.0
#		else:
#			my_dataset[x]['ratio_sal_totalp'] = num(my_dataset[x]['salary'])/num(my_dataset[x]['total_payments'])
#	if math.isnan(num(my_dataset[x]['to_messages'])) | (num(my_dataset[x]['to_messages']) == 0.0):
#		my_dataset[x]['to2from_poi_to_this_person'] = 0.0
#	else:
#		my_dataset[x]['to2from_poi_to_this_person'] = num(my_dataset[x]['from_poi_to_this_person'])/num(my_dataset[x]['to_messages'])
#	if math.isnan(num(my_dataset[x]['from_messages'])) | (num(my_dataset[x]['from_messages']) == 0.0):
#		my_dataset[x]['from2from_this_person_to_poi'] = 0.0
#	else:
#		my_dataset[x]['from2from_this_person_to_poi'] = num(my_dataset[x]['from_this_person_to_poi'])/num(my_dataset[x]['from_messages'])

for x in my_dataset:
	if math.isnan(num(my_dataset[x]['total_payments'])):
		my_dataset[x]['ratio_exso_totalp'] = 0.0
	else:
		if num(my_dataset[x]['total_payments']) == 0.0:
			my_dataset[x]['ratio_exso_totalp'] = 0.0
		else:
			my_dataset[x]['ratio_exso_totalp'] = num(my_dataset[x]['exercised_stock_options'])/num(my_dataset[x]['total_payments'])
	if math.isnan(num(my_dataset[x]['to_messages'])) | (num(my_dataset[x]['to_messages']) == 0.0):
		my_dataset[x]['to2from_poi_to_this_person'] = 0.0
	else:
		my_dataset[x]['to2from_poi_to_this_person'] = num(my_dataset[x]['from_poi_to_this_person'])/num(my_dataset[x]['to_messages'])
	if math.isnan(num(my_dataset[x]['from_messages'])) | (num(my_dataset[x]['from_messages']) == 0.0):
		my_dataset[x]['from2from_this_person_to_poi'] = 0.0
	else:
		my_dataset[x]['from2from_this_person_to_poi'] = num(my_dataset[x]['from_this_person_to_poi'])/num(my_dataset[x]['from_messages'])

#features_list.append("ratio_sal_totalp")
#features_list.append("ratio_bonus_totalp")
features_list.append("ratio_exso_totalp")
features_list.append("to2from_poi_to_this_person")
features_list.append("from2from_this_person_to_poi")
print "Final features list:", features_list
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
from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

#from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values=0, strategy='mean', axis=0)
#imp.fit(features)
#features = imp.transform(features)

from sklearn.svm import SVC
#clf = SVC(verbose=True)

from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=10)

#clf = DecisionTreeClassifier(random_state=0, max_depth=3)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

#estimators = [('reduce_dim', PCA()), ('decisiontree', DecisionTreeClassifier(random_state=0, max_depth=3, min_samples_leaf=2))] # precision 0.42186 recall 0.24700
#estimators = [('reduce_dim', PCA()), ('decisiontree', DecisionTreeClassifier(random_state=0, min_samples_leaf=1))] # precision 0.29493 recall 0.30275
#estimators = [('reduce_dim', PCA()), ('decisiontree', DecisionTreeClassifier(random_state=0, min_samples_leaf=1, max_features='auto'))] # precision 0.27664 recall 0.28950
#estimators = [('reduce_dim', PCA(n_components=4)), ('classifier', RandomForestClassifier(n_estimators=120, min_samples_leaf=2))] # Accuracy: 0.85760       Precision: 0.43573      Recall: 0.23050 F1: 0.30150     F2: 0.25447
#estimators = [('reduce_dim', PCA(n_components=3)), ('classifier', SVC())] # Precision or recall may be undefined due to a lack of true positive predicitons.
#estimators = [('reduce_dim', PCA(n_components=4)), ('classifier', GaussianNB())] # Accuracy: 0.85953       Precision: 0.45668      Recall: 0.28200 F1: 0.34869     F2: 0.30536
#estimators = [('reduce_dim', PCA(n_components=4)), ('classifier', AdaBoostClassifier(n_estimators=100))] #Accuracy: 0.83440       Precision: 0.35765      Recall: 0.30400 F1: 0.32865     F2: 0.31340
estimators = [('reduce_dim', PCA(n_components=3)), ('classifier', KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree'))] #Accuracy: 0.87220       Precision: 0.53496      Recall: 0.31750 F1: 0.39849     F2: 0.34560

clf = Pipeline(estimators)

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

from sklearn.cross_validation import cross_val_score

clf.fit(features_train, labels_train)

#dt = clf.named_steps['decisiontree']
#print "Feature importances:", dt.feature_importances_

pca = clf.named_steps['reduce_dim']
print("PCA Explained: ", pca.explained_variance_ratio_) 
print pca.components_ 

scores = cross_val_score(clf, features_test, labels_test)
print "Cross-validation scores mean: ", scores.mean()
print "Classifier score: ", clf.score(features_test, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)