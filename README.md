# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the necessary python libraries.
2. Introduce the variables needed to execute the function.
3. Using for loop apply the concept using the formulae.
4. Execute the program and plot the graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: MANO 
RegisterNumber: 212221040100
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,1000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
def computeCost(X,y,theta):
  m=len(y) #length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2
  return 1/(2*m) * np.sum(square_err) #returning ] 
  data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function
def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions -y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```




## Output:
1.Profit prediction graph

![Profit prediction graph](https://user-images.githubusercontent.com/128135126/229297745-35008d99-cc02-477b-a097-5efbcb0a30c8.png)

2.Compute cost value

![Compute cost value](https://user-images.githubusercontent.com/128135126/229297766-cdde2e4f-cf10-46d6-88b6-b8f1f04ab4d8.png)

3.h(x) value

![h(x) value](https://user-images.githubusercontent.com/128135126/229297777-5fcdcd99-0954-4aa3-9ae7-151cb808ff65.png)

4.Cost function using Gradient Descent graph

![Cost function using gradient descent graph](https://user-images.githubusercontent.com/128135126/229297805-9aec7927-1b4a-4631-a2dc-d0aec4b9ca50.png)

5.Profit prediction graph

![Profit prediction](https://user-images.githubusercontent.com/128135126/229297827-9f401d64-2648-4687-b7d6-1daefcd6bcc4.png)

6.Profit for the population 35,000

![Profit for the population 35000](https://user-images.githubusercontent.com/128135126/229297847-791ec19e-f63b-4c5d-b2d7-c497b05cf491.png)

7.Profit for the population 70,000

![Profit for the population 70000](https://user-images.githubusercontent.com/128135126/229297865-0cae4b12-9cbe-4d0b-bd4c-813e902f1842.png)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
