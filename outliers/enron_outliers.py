#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("L:/project/ud120-projects-master/tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("L:/project/ud120-projects-master/final_project/final_project_dataset.pkl", "r") )

#print data_dict["TOTAL"]
errorvalue=data_dict.pop("TOTAL")
#print errorvalue    
#print data_dict
       
for name in data_dict:
    if data_dict[name]["bonus"]==7000000.0:
        print name
    
    
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
print data


### your code below
salary1=[]
bonus1=[]
for point in data:
    salary = point[0]
    bonus = point[1]
    salary1.append(salary)
    bonus1.append(bonus)
    matplotlib.pyplot.scatter( salary, bonus )

bonus1.sort() 
print bonus1

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()