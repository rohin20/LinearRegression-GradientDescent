import numpy as np
import pandas as pd

data = pd.read_csv("californiahousing.csv")


numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Convert categorical column to numerical (one-hot encoding)
data = pd.get_dummies(data, columns=['ocean_proximity'])

# Separate features and target variable
X = data.drop(columns="median_house_value")
y = data['median_house_value']

# Standardize the numeric features
numeric_columns = X.select_dtypes(include=[np.number]).columns
X[numeric_columns] = (X[numeric_columns] - X[numeric_columns].mean()) / X[numeric_columns].std()

m = len(y)
X_b = np.c_[np.ones((m, 1)), X]
#first column is made with just 1s

np.random.seed(42)
theta = np.random.randn(X_b.shape[1], 1) #vector of random thetas
y = y.values.reshape(-1, 1) #change to column vector

stepSize = 0.01

def prediction(X, theta):
    return X.dot(theta)

def costFunction(X,y,theta):
    ypred = prediction(X,theta)
    cost = (1/(2*m)) * np.sum((ypred-y)**2)
    return cost

for i in range(1000):
    ypred = prediction(X_b,theta)
    cost = costFunction(X_b,y,theta)
    dcost = (1/m) * np.matmul(X_b.transpose(),(ypred-y))
    theta = theta-stepSize*dcost

ypred = prediction(X_b, theta)
mse = np.mean((y-ypred)**2)

print(f"Mean Squared error = {mse}")