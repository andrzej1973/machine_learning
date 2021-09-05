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
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

############################################################################################
# Linear regression with one variable                                                      #
# based on Coursera ML Course                                                              #
#                                                                                          #
# Problem:                                                                                 #
# Implement linear regression with one variable to predict profits for a food truck.       #
# Suppose you are the CEO of a restaurant franchise and are considering different          #
# cities for opening a new outlet. The chain already has trucks in various cities and      #
# you have data for profits and populations from the cities.                               #
# You would like to use this data to help you select which city to expand to next.         #
############################################################################################

# Importing the dataset
data = pnd.read_csv("/home/pi/PyScripts/ML/data/CityPopulation_RevePerTrack.txt")

print("Dimensions of loaded dataset: " + str(data.shape))

###########################################################################################
# X - vector including 97 values of single feature, which is size of city population      #
# y - vector including 97 values of revenues per food track operating in the city with    #
#     given population                                                                    #
###########################################################################################

X = data[["CityPopulation"]]
y = data[["RevenuesPerTrack"]]


#Fitting Simple Linear Regression to the set

############################################################################################                                  
#  our linear model: y = w1*x1 + w0                                                        #
#                    where:                                                                #
#                           w1 = coef_                                                     #
#                           w0 = intercpet_                                                #
#                                                                                          #
# algorithms description:                                                                  #
#         https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares #
#                                                                                          #
############################################################################################


reg_linreg = linear_model.LinearRegression()
regularization_param = 0.5
reg_lasso = linear_model.Lasso(alpha=regularization_param)

reg_linreg.fit(X,y)
reg_lasso.fit(X,y)

print ("y = coef_*x + intercept_")
print ("parameters calculated by LinearRegression algorithm:")
print("coef_: " + str(reg_linreg.coef_))
print("intercept_: " + str(reg_linreg.intercept_))
print ("parameters calculated by Lasso algorithm:")
print("coef_: " + str(reg_lasso.coef_))
print("intercept_: " + str(reg_lasso.intercept_))

#Predicting the set results

y_pred_linreg = reg_linreg.predict(X)
y_pred_lasso = reg_lasso.predict(X)

#Calculate mean square error 

print("Mean square error for linear regression algorithm: " + str(mean_squared_error(y,y_pred_linreg)))
print("Mean square error for lasso algorithm (regularization parameter alpha = " + str(regularization_param) + "): " + str(mean_squared_error(y,y_pred_lasso)))

city_population = 16
print("Predicted profit per food track for the city with population of " + str(city_population) + " 000 is (linear regression): " + str(reg_linreg.predict([[city_population]])))
print("Predicted profit per food track for the city with population of " + str(city_population) + " 000 is (lasso regression): " + str(reg_lasso.predict([[city_population]])))

#Visualising the set results
plt.scatter(X, y, color = 'red', marker = 'x', label = 'Training Data')
plt.plot(X, y_pred_linreg, color = 'blue', label = 'Linear Regression')
plt.plot(X, y_pred_lasso, color = 'green', label = 'Lasso')
plt.legend(loc = 'upper right')
plt.title('City Population vs. Profit Per Track')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit Per Track in $10,000s')
plt.legend()
plt.show()