# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:37:13 2022

@author: Revathi
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import dataset
dataset = pd.read_csv('Data/Classification/Apply_Job.csv')

X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# building our decision tree classifier and fitting the model
from sklearn.tree import DecisionTreeClassifier
dt_c = DecisionTreeClassifier()
dt_c.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

pred_train = dt_c.predict(X_train)
pred_test = dt_c.predict(X_test)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
import numpy as np
#Define Variables
clf = dt_c
h = 0.01
X_plot, z_plot = X, y 

#Standard Template to draw graph
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh
Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,
             alpha = 0.7, cmap = ListedColormap(('blue', 'red')))


for i, j in enumerate(np.unique(z_plot)):
    plt.scatter(X_plot[z_plot == j, 0], X_plot[z_plot == j, 1],
                c = ['blue', 'red'][i], cmap = ListedColormap(('blue', 'red')), label = j)
   #X[:, 0], X[:, 1] 
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Decision Tree Classification')
plt.xlabel('Experience in Years')
plt.ylabel('Salary in lakhs')
plt.legend()

plt.show()
train_accuracy = accuracy_score(y_train, pred_train)
test_accuracy = accuracy_score(y_test, pred_test)

print(train_accuracy)
print(test_accuracy)

