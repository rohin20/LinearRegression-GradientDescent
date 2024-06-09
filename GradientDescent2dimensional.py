import numpy as np
import matplotlib.pyplot as plt

#generate and plot sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Sample Data")
plt.show()

m = len(y)
print(m)
theta0 =  np.random.randn()
theta1 =  np.random.randn()
# theta0 and theta1 are random so that we can converge to the min from any point
stepSize = 0.01

def prediction(X,theta0,theta1):
    return (theta1*X + theta0)

def costFunction(X,y,theta0,theta1):
    ypred = prediction(X,theta0,theta1)
    return (1/(2*m))*np.sum((ypred-y)**2)

for i in range(1000):
    ypred = prediction(X,theta0,theta1)
    dtheta0 = (1/m)*np.sum((ypred-y))
    dtheta1 = (1/m)*np.sum((ypred-y)*X)
    theta0 =theta0 - stepSize*dtheta0
    theta1 =theta1 - stepSize*dtheta1

plt.scatter(X, y, label="Data")
plt.plot(X,prediction(X,theta0,theta1), color = "red", label = "linear regression") 
plt.title("data")   
plt.show()



