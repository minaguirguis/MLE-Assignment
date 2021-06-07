#Mina Guirguis
#CS 482
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random 


dataSet = [4, 5, 7, 8, 8, 9, 10, 5, 2, 3, 5, 4, 8, 9]

num = sum(dataSet) / len(dataSet) 
ran1 = random.choice(dataSet)
ran2 = random.choice(dataSet)


print("First num:",ran1,"First sd:",ran2)

def compare_data_to_dist(x1, mu_1, sd_1,learning_rate,interations):
    ll_1 = 0
    ll_2=0
    gradient2=0

    for i in x1:
        ll_1 += np.log(norm.pdf(i, mu_1, sd_1))
        ll_2+=i
        gradient2+=(i-num)**2

    print(gradient2)
    print("The LL of x1 for num = %d and sd = %d is: %.4f" % (mu_1, sd_1, ll_1))
    ll_2=ll_2/len(dataSet)

    gradient2=gradient2/len(dataSet)
    mu_1 = mu_1 - learning_rate*ll_2

    sd_1 = sd_1 - learning_rate*gradient2

 
    for i in range(interations):
        interations=interations - 1
        compare_data_to_dist(dataSet,mu_1,sd_1,0.2,interations)
        break

compare_data_to_dist(dataSet, mu_1 = 8, sd_1 = 5, learning_rate= 0.1, interations = 10)


x1 = np.array([[8, 16, 22, 33, 50, 51]]).T
y1 = np.array([[5, 20, 14, 32, 42, 58]]).T

from sklearn import preprocessing

x2 = preprocessing.StandardScaler().fit(x1)
y2 = preprocessing.StandardScaler().fit(y1)

hold1 = len(x1)

x3 = np.c_[np.ones((hold1, 1)), x1]
print(x3)

x3 = x2.transform(x3)
print(x3)

y1 = y2.transform(y1)
print(y1)

mle = np.linalg.inv(x3.T.dot(x3)).dot(x3.T).dot(y1)
print('mle = ', mle)
plt.plot(x3[:,1], y1, "b.")
x5 = np.array([[-2], [2]])

x6 = np.c_[np.ones((2, 1)), x5]
y7 = x6.dot(mle)

plt.plot(x5, y7) 

def gradientUp(p, eta, w_path=None):
    hold1 = len(x3)
    plt.plot(x3[:,1], y1, "b.")
    nIt = 1000
    for itrs in range(nIt):
        if itrs < 10:
            y7 = x6.dot(p)
            changeUp = "b-" if itrs > 0 else "r--"
            plt.plot(x5, y7, changeUp)
        grad = 2/hold1*x3.T.dot(x3.dot(p)-y1)

        p = p - eta * grad
        if w_path is not None:
            w_path.append(p)
    plt.xlabel("$x1$", fontsize=18)
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

np.random.seed(42) 
p = np.random.randn(2,1)  
print('Initial p = ', p)
w_path_bgd = []
plt.figure(figsize=(10,4)) 
gradientUp(p, eta=0.1)

plt.ylabel("$y1$", rotation=0, fontsize=18)
plt.show()

def gradientUp(p, eta, w_path=None):
    hold1 = len(x3)
    plt.plot(x3[:,1], y1, "b.")
    nIt = 1000 
    for itrs in range(nIt):
        if itrs < 10:
            y7 = x6.dot(p)
            changeUp = "b-" if itrs > 0 else "r--"
            plt.plot(x5, y7, changeUp)
            random_index=np.random.randint(hold1) 
            xi=x3[random_index:random_åindex+1] 
            yi=y1[random_index:random_index+1] 

        grad = 2*xi.T.dot(xi.dot(p)-yi) 
        p = p - eta * grad
        if w_path is not None:
            w_path.append(p)
    plt.xlabel("$x1$", fontsize=18)
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

np.random.seed(42) 
p = np.random.randn(2,1)  


w_path_bgd = []
plt.figure(figsize=(10,4)) 
gradientUp(p, eta=0.1)

plt.ylabel("$y1$", rotation=0, fontsize=18)
plt.show()
