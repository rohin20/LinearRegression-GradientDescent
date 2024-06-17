import numpy as np
import pandas as pd

data = pd.read_csv("californiahousing.csv")
data = data.fillna(data.median())

X = data.drop(columns="median_house_value")
y = data['median_house_value']

#standardize
Xstandardized = (X-(X.mean()))/(X.std())

m = len(y)
X_b = np.c_[np.ones((m, 1)), Xstandardized]
#first column is made with just 1s

np.random.seed(42)
theta = np.random.randn(X_b.shape[1], 1) #vector of random thetas
y = y.values.reshape(-1, 1) #change to column vector

stepSize = 0.01

def prediction(X, theta):
    return X.dot(theta)

def costFunction(X,y,theta):
    ypred = prediction(X,theta)
    cost = (1/2*m) * np.sum((ypred-y)**2)
    return cost

for i in range(1000):
    ypred = prediction(X,theta)
    cost = costFunction(X,y,theta)
    dcost = (1/m) * np.matmul(X.transpose(),(ypred-y))
    theta = theta-stepSize*dcost
