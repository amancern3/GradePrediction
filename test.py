#Linear Regression implementation

import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

 # most important part of machine learning is trimming the data set

data = pd.read_csv("student-mat.csv", sep = ";")

print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"   #known as a label -- thing we are training our dataset to predict

X = np.array(data.drop([predict], 1))   # our training data -- dropping the G3 value
y = np.array(data[predict])

x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
linear.fit(x_test, y_test)
acc=linear.score(x_test, y_test)
print(acc)

print("Co: \n" , linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])
