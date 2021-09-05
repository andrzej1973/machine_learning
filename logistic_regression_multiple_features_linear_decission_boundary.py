#!/usr/bin/python3

#Do not use pip3 to install numpy and matplotlib on RB!
#instead use following two commands:
#sudo apt install python3-numpy
#sudo apt install python3-matplotlib
#https://www.geeksforgeeks.org/data-visualization-with-python-seaborn/
#sudo apt install python3-seaborn
#sudo apt install python3-pandas
#sudo apt install python3-sklearn

import pandas as pnd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score


############################################################################################
# Logistic regression with multiple variables (features)                                   #
# based on Coursera ML Course                                                              #
#                                                                                          #
# Problem:                                                                                 #
# Predict whether a student gets admitted into a university based on their                 #
# results on two exams. Historical data from previous applicants is available and that can #
# be uses as a training set for logistic regression.                                       #
############################################################################################

# Importing the dataset

###########################################################################################
# Input data in StudentExamResultsAndAdmissionDecission.txt                               #
# Column 1: Result of Exam 1                                                              #
# Column 2: Result of Exam 2                                                              #
# Column 3: Admission Decission                                                           #
###########################################################################################

dataframe = pnd.read_csv("/home/pi/PyScripts/ML/data/StudentExamResultsAndAdmissionDecission.txt")

print("Dimensions of loaded dataset: " + str(dataframe.shape))

#Convert pandas dataframe to numpy array
data = dataframe.values

#Split dataset into features array X and results vectory y
X = data[:,[0,1]]
y = data[:,2]

#Fitting Logistic Regression to the set

reg = linear_model.LogisticRegression(solver = 'lbfgs', max_iter = 400)
reg.fit(X, y)

#Retrieve model parameters
b = reg.intercept_[0]
w1, w2 = reg.coef_.T
print ("Decission boundary parameters:")
print ("y = 1 / (1 + exp (-z)), where:")
print ("z = w1*x1 + w2*x2 + b")
print("coef_: " + str(w1) + " " + str(w2))
print("intercept_: " + str(b))

#Calculate the intercept and gradient of the decission boundary
#A decision boundary is the region of a problem space in which the output label of a classifier is ambiguous
#Decission boundary shows for which set of parameters y = 1/2

####################################
#                                  #
#  y = 1 / ( 1 + exp ( -z ) )      #
#  y = 1 / 2 => z = 0              #
#                                  #
#  z = w1*x1 + w2*x2 + b           #
#                                  #
#  -w2x2 = w1x1 + b - z            #
#                                  #
#   x2 = m * x1 + c                #
#                                  #
#   m = - w/w2                     #
#   c = - b/w2                     #
#                                  #
####################################
c =  -b / w2
m = -w1 / w2


#Predicting Test Result
#Predict Admission Result for particular feature set (combination of Exam1 and Exam2 results)
Exam1_result = 45
Exam2_result = 85

X_predict=[[Exam1_result,Exam2_result]]

admission_predict = reg.predict(X_predict)

print("Admission result prediction is:" + str(admission_predict) + " for the following exam results: Exam1=" + str(Exam1_result) + " and Exam2=" + str(Exam2_result))

#Visualising the set results

#Find Indices of Positive and Negative Examples
#for where to work correctly make sure dataframe has column names included in first raw!
pos = np.where(data[:,2] == 1)
neg  = np.where(data[:,2] == 0)

Ex1min = np.min(X[:,0])
Ex1max = np.max(X[:,0])

Ex1 = np.array([Ex1min,Ex1max])

Ex2 = m*Ex1 + c

plt.plot(Ex1,Ex2, color = 'green')
plt.scatter(X[pos,0], X[pos,1], color = 'black', marker = '+', label = 'Admitted')
plt.scatter(X[neg,0], X[neg,1], color = 'yellow', marker = 'o', label = 'Not Admitted')
#plt.plot(X, y_pred_linreg, color = 'blue', label = 'Linear Regression')
#plt.plot(X, y_pred_lasso, color = 'green', label = 'Lasso')
plt.legend(loc = 'upper right')
plt.title('Results of Exam 1 and Exam 2 vs. Admission Result')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
plt.legend()
plt.show()