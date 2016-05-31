#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
import math
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

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
#features_list = ['poi', 'salary', 'bonus'] # You will need to use more features
features_list = ['poi', 'salary', 'bonus', 'total_payments', 'exercised_stock_options'] # You will need to use more features

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

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print "Number of data points: ", len(data_dict)

### Task 2: Remove outliers
#Remove "TOTAL" line
data_dict.pop( "TOTAL", 0 )
data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0 )

outliers = []

print "Data revalidation:"
## Data re-validation
for key1 in data_dict:
	if ( num(data_dict[key1]['total_payments']) <> (num(data_dict[key1]['salary']) + num(data_dict[key1]['bonus']) +
	num(data_dict[key1]['long_term_incentive']) + num(data_dict[key1]['deferred_income']) + num(data_dict[key1]['deferral_payments']) +
	num(data_dict[key1]['loan_advances']) + num(data_dict[key1]['other']) + num(data_dict[key1]['expenses']) + num(data_dict[key1]['director_fees'])
	) 
	):
		print "Inconsistent financial data: ", key1
		outliers.append(key1)
	if ( num(data_dict[key1]['total_stock_value']) <> (num(data_dict[key1]['exercised_stock_options']) + num(data_dict[key1]['restricted_stock']) +
	num(data_dict[key1]['restricted_stock_deferred'])
	) 
	):
		print "Inconsistent stock data: ", key1
		outliers.append(key1)
	if (data_dict[key1]['total_payments'] == 'NaN') & (data_dict[key1]['total_stock_value'] == 'NaN'):
		print 'No data at all ', key1
		outliers.append(key1)
		
for o in outliers:
	data_dict.pop( o, 0 ) #No fin data at all

print "Data points after outliers removal:", len(data_dict)

## Let's produce some stats on the data set
dict_nan_counts = {}
dict_nan_pp_counts = {}
for key1 in data_dict:
	dict_nan_pp_counts[key1] = 0
	for key2 in data_dict[key1]:
		#print key2
		if data_dict[key1][key2] == 'NaN':
			dict_nan_pp_counts[key1] = dict_nan_pp_counts[key1] + 1
			if key2 in dict_nan_counts:
				dict_nan_counts[key2] = dict_nan_counts[key2] + 1
			else:
				dict_nan_counts[key2] = 1
	
print "Features NaN counts: ", dict_nan_counts
print "Biggest number of NaNs per person:"
for key1 in dict_nan_pp_counts:
	if dict_nan_pp_counts[key1] > 15:
		print key1, ' ', dict_nan_pp_counts[key1]
#print dict_nan_pp_counts

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

##Let create a new feature: ratio of exercised_stock_options to total payments/total stocks
## Ration of from_poi_to_this_person to to_messages
## Ration of from_this_person_to_poi to from_messages
for x in my_dataset:
	if math.isnan(num(my_dataset[x]['total_payments'])):
		my_dataset[x]['ratio_bonus_totalp'] = 0.0
	else:
		if num(my_dataset[x]['total_payments']) == 0.0:
			my_dataset[x]['ratio_bonus_totalp'] = 0.0
		else:
			my_dataset[x]['ratio_bonus_totalp'] = num(my_dataset[x]['bonus'])/num(my_dataset[x]['total_payments'])
	#if math.isnan(num(my_dataset[x]['to_messages'])) | (num(my_dataset[x]['to_messages']) == 0.0):
	#	my_dataset[x]['to2from_poi_to_this_person'] = 0.0
	#else:
	#	my_dataset[x]['to2from_poi_to_this_person'] = num(my_dataset[x]['from_poi_to_this_person'])/num(my_dataset[x]['to_messages'])
	#if math.isnan(num(my_dataset[x]['from_messages'])) | (num(my_dataset[x]['from_messages']) == 0.0):
	#	my_dataset[x]['from2from_this_person_to_poi'] = 0.0
	#else:
	#	my_dataset[x]['from2from_this_person_to_poi'] = num(my_dataset[x]['from_this_person_to_poi'])/num(my_dataset[x]['from_messages'])
		
	if math.isnan(num(my_dataset[x]['total_stock_value'])):
		my_dataset[x]['ratio_exso_totals'] = 0.0
	else:
		if num(my_dataset[x]['total_stock_value']) == 0.0:
			my_dataset[x]['ratio_exso_totals'] = 0.0
		else:
			my_dataset[x]['ratio_exso_totals'] = num(my_dataset[x]['exercised_stock_options'])/num(my_dataset[x]['total_stock_value'])
	
features_list.append("ratio_bonus_totalp")
features_list.append("ratio_exso_totals")
#features_list.append("to2from_poi_to_this_person")
#features_list.append("from2from_this_person_to_poi")
print "Final features list:", features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

count_poi = 0
count_nonpoi = 0
count_all = len(data)

print "All: ", count_all

#for point in data:
#    salary = point[1]
#    bonus = point[2]
#    tpayments = point[3]
#    exoptions = point[4]
#    sal_total = point[7]
#    if float(point[0]) == 0.0 :
#    	# Non POI
#	#matplotlib.pyplot.scatter( salary, tpayments, c='white' )
#	matplotlib.pyplot.scatter( salary, bonus, c='red' )
#	#matplotlib.pyplot.scatter( salary, sal_total, c='black' )
#	#print salary,':', sal_total
#	count_nonpoi = count_nonpoi + 1
#    	#matplotlib.pyplot.scatter( salary, director_fee, c='black' )
#    else:
#    	#POI
#	count_poi = count_poi + 1
#	#matplotlib.pyplot.scatter( salary, tpayments, c='blue' )
#	matplotlib.pyplot.scatter( salary, bonus, c='yellow' )
#	#matplotlib.pyplot.scatter( salary, sal_total, c='red' )
#    	#matplotlib.pyplot.scatter( salary, director_fee, c='yellow' )
#
#matplotlib.pyplot.xlabel("salary")
#matplotlib.pyplot.ylabel("bonus")
##matplotlib.pyplot.show()

print "POIs: ", count_poi
print "NON-POIs: ", count_nonpoi

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.pipeline import Pipeline

#from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 #Requried for SelectKBest
from sklearn.feature_selection import f_classif #Requried for SelectKBest

# Provided to give you a starting point. Try a variety of classifiers.
#clf = GaussianNB()
#clf = SVC(verbose=True)
#clf = RandomForestClassifier(n_estimators=10)
#clf = DecisionTreeClassifier(random_state=0, max_depth=3)

##Gaussian NB
#classifier = GaussianNB()
##Accuracy: 0.84929       Precision: 0.45439      Recall: 0.27400 F1: 0.34186     F2: 0.29763
#params=dict(scaler__feature_range=[(0,1)],
#		reduce_dim__k=[3,4,5,6],
#		reduce_dim__score_func=[chi2, f_classif])

##DecisionTreeClassifier
#classifier = DecisionTreeClassifier()
##Accuracy: 0.80150       Precision: 0.31566      Recall: 0.33350 F1: 0.32434     F2: 0.32977
#params = dict(classifier__criterion =['gini', 'entropy'], classifier__min_samples_split = [1, 2, 3, 4],
#		classifier__min_samples_leaf = [1,2,3], classifier__max_depth = [None,1,2,3,4,5],
#		scaler__feature_range=[(0,1)],
#		reduce_dim__k=[2,3,4,5,6],
#		reduce_dim__score_func=[chi2, f_classif])

##AdaBoost
#classifier = AdaBoostClassifier()
##Accuracy: 0.82564       Precision: 0.35111      Recall: 0.26000 F1: 0.29876     F2: 0.27423
#params = dict(classifier__n_estimators =[100, 500, 1000], classifier__learning_rate = [0.5, 1], scaler__feature_range=[(0,1)],
#		reduce_dim__k=[3,4,5,6],
#		reduce_dim__score_func=[chi2, f_classif])

##RandomForest
#classifier = RandomForestClassifier()
##Accuracy: 0.87171       Precision: 0.59273      Recall: 0.32600 F1: 0.42065     F2: 0.35824
#params = dict(	classifier__n_estimators=[100, 120, 200],
#		classifier__min_samples_leaf =[1, 2, 3],
#		classifier__max_depth=[3, 5, None],
##		scaler__feature_range=[(0,1),(-1, 1)],
#		scaler__feature_range=[(0,1)],
##		reduce_dim__n_components=[3,4,8]
#		reduce_dim__k=[3,4,5,6],
#		reduce_dim__score_func=[chi2, f_classif])

##SVC
#classifier = SVC()
##Accuracy: 0.81629       Precision: 0.32750      Recall: 0.27150 F1: 0.29688     F2: 0.28111
#params = dict(	classifier__C =[10000, 5000, 1000],
#		classifier__kernel = ['rbf', 'sigmoid', 'poly'],
#		classifier__gamma = [0, 1, 2, 5, 7],
#		classifier__coef0 =[0, 0.5, 1, 2],
##		scaler__feature_range=[(0,1),(-1, 1)],
#		scaler__feature_range=[(0,1)],
#		reduce_dim__k=[3,4,5,6],
#		reduce_dim__score_func=[chi2, f_classif])

##KNeighborsClassifier
classifier = KNeighborsClassifier()
##Accuracy: 0.87071       Precision: 0.58979      Recall: 0.31200 F1: 0.40811     F2: 0.34445
params = dict(classifier__n_neighbors=[2,3,5], classifier__p=[1,2,3,4,5],scaler__feature_range=[(0,1)],
		reduce_dim__k=[3,4,5,6],
		reduce_dim__score_func=[chi2, f_classif])

estimators = [('scaler', MinMaxScaler()),('reduce_dim', SelectKBest()), ('classifier', classifier)]
clf = Pipeline(estimators)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

cv = StratifiedShuffleSplit(y = labels, 
                            n_iter = 10, 
                            test_size = 0.1, 
                            random_state = 42)

scores = ['recall'] #['precision', 'recall', 'f1']

for score in scores:
	print '--', score, '--'
	grid_search = GridSearchCV(clf, 
                           param_grid = params, 
                           verbose = 1,
                           cv = cv,
                           scoring=score)
	grid_search.fit(features, labels)
	print "(grid_search.best_estimator_.steps): ", (grid_search.best_estimator_.steps)
	print "(grid_search.best_score_): ", (grid_search.best_score_)
	print "(grid_search.best_params_): ", (grid_search.best_params_)
	print "(grid_search.scorer_): ", (grid_search.scorer_)
	
	clf.set_params(**grid_search.best_params_)
	
	print 'SelectKBest scores: ', clf.named_steps['reduce_dim'].fit(features_train, labels_train).scores_
	print 'SelectKBest scores: ', clf.named_steps['reduce_dim'].fit(features_train, labels_train).get_params()
		
clf.fit(features, labels)

if isinstance(clf.named_steps['classifier'], DecisionTreeClassifier):
	print 'Features importances: ', clf.named_steps['classifier'].feature_importances_

#dt = clf.named_steps['decisiontree']
#print "Feature importances:", dt.feature_importances_

#pca = clf.named_steps['reduce_dim']
#print("PCA Explained: ", pca.explained_variance_ratio_) 
#print pca.components_ 

#print features_train

scores = cross_val_score(clf, features, labels)
print "Cross-validation scores mean: ", scores.mean()
print "Classifier score: ", clf.score(features_test, labels_test)

print test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)