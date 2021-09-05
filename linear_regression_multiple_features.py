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
# Linear regression with multiple variables (features)                                     #
# based on Coursera ML Course                                                              #
#                                                                                          #
# Problem:                                                                                 #
# Implement linear regression with two variables to predict house market prices based on   #
# size of the house (in square feet), and number of bedrooms                               #
############################################################################################

# Importing the dataset

###########################################################################################
# Input data in HouseSqFeet_NoOfBedrooms_Price.txt                                        #
# Column 1: Size of the house (in square feet)
# Column 2: Number of Bedrooms
# Column 3: Price of the house in $
###########################################################################################

dataframe = pnd.read_csv("/home/pi/PyScripts/ML/data/HouseSqFeet_NoOfBedrooms_Price.txt")

print("Dimensions of loaded dataset: " + str(dataframe.shape))

#Convert pandas dataframe to numpy array
data = dataframe.values

#Split dataset into features array X and results vectory y
X = data[:,[0,1]]
y = data[:,2]

#Regularize input parameters/features
#  Regularization method (feature scaling) used:
#  X_normalized = (X - XMean)/StdDevX
#  Remember that when predicting values you also need to use same mean and standard
#  deviation numbers as those used during fitting the model

scaler = StandardScaler()  
scaler.fit(X)  
X_normalized = scaler.transform(X)

print ("Scaler parameters for House Size in Sq. Feet and Number of Bedrooms features:")
print("Mean Values:" + str(scaler.mean_))
print("Standard Deviations: " + str(np.sqrt(scaler.var_)))

#Fitting Simple Linear Regression to the set

############################################################################################                                  
#  our linear model: y = w1*x1 + w2*x2 + w0                                                #
#                    where:                                                                #
#                           w1, w2 = coef_                                                 #
#                           w0 = intercpet_                                                #
#                                                                                          #
# algorithms description:                                                                  #
#         https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares #
#                                                                                          #
############################################################################################

regularization_param = 0.1
reg = linear_model.Lasso(alpha=regularization_param)

reg.fit(X_normalized,y)
reg.fit(X_normalized,y)

print ("y = coef_(1)*x1 + coef_(2)*x2 + intercept_")
print ("parameters calculated by Lasso algorithm:")
print("coef_: " + str(reg.coef_))
print("intercept_: " + str(reg.intercept_))

#Predicting the set results
y_pred = reg.predict(X_normalized)

#Calculate mean square error 
print("Mean square error for lasso algorithm (regularization parameter alpha = " + str(regularization_param) + "): " + str(mean_squared_error(y,y_pred)))

#Predict house price for particular feature set, which was not part of training set
house_size = 1650
house_no_of_bedrooms = 3

#Normalize features you want predict the result for
X_predict=[[house_size,house_no_of_bedrooms]]
X_predict_normalized = scaler.transform(X_predict)

print("Predicted price of " + str(house_size) + "sq. feet/" + str(house_no_of_bedrooms) + " bedroom house located in Portland, Oregon is:" + str(reg.predict(X_predict_normalized)) + "$")

#Visualising the set results
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(house_size,house_no_of_bedrooms,reg.predict(X_predict_normalized) , color = 'red')
ax.scatter3D(X[:,0],X[:,1], y, color = 'blue')
ax.set_title('House Price as a Function of Size and Number of Bedrooms')
ax.set_xlabel('Size [sq. feet]')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price [$]')
plt.show()
