#To run the code, open the pyhton file(GYM) and the csv file(Dataset_GYM) as a folder on vs code.

#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Importing the dataset
Dataset =  pd.read_csv('Dataset_GYM.csv') 
X = Dataset.iloc[:, [0, 1]].values
y = Dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

# Scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric =  'manhattan', p = 2)  # Manhattan distance = |x1 - x2| + |y1 - y2|
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Creating confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy = %f'% accuracy)

# Calculating precision
precision = precision_score(y_test, y_pred)
print('Precision = %f' % precision)

# Calculating recall
recall = recall_score(y_test, y_pred)
print('Recall = %f' % recall)

# Function made which shows the colormap (which is converted from the data)
def Graph(argument1, arguement2):
  from matplotlib.colors import ListedColormap
  X_set, y_set = argument1, arguement2
  X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01),np.arange(start=X_set[:, 1].min() - 1, stop =X_set[:, 1].max() + 1, step = 0.01))
  plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha= 0.75, cmap = ListedColormap(('red', 'blue')))

  plt.xlim(X1.min(), X1.max())
  plt.ylim(X2.min(), X2.max())
  for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label=j)
  plt.xlabel('Weight')
  plt.ylabel('Age')
  plt.legend()
  plt.show()
  
Graph(X_train, y_train)  #Visualizing the Training set results
Graph(X_test, y_test)    #Visualizing the Test set results