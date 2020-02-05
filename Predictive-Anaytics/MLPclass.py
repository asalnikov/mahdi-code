# MLPclassifier

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing


# Importing the dataset


#file = pd.read_csv('statistics8.csv', sep=',')

file = pd.read_csv('midClass.csv', sep=',', header= 0)

le = preprocessing.LabelEncoder()
file = file.apply(le.fit_transform)


df = pd.DataFrame(file)

'''
while True:
    if df.user:
        print(df.user)
'''



#y = df.state
#X = df.drop('state', axis=1)

#X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].values
#y = df.iloc[:, 3].values

#X = df.iloc[:, 0:9].values

#X = df.iloc[:, [0, 12]].values
#y = df.iloc[:, -1].values

print("DFFFFFFFFFFFFFFFFFF",df)

#X = X.as_matrix().astype(np.float)
#y = y.as_matrix().astype(np.float)


#X = df.['name','id']

y = file.state
X = file.drop('state', axis=1)


# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#print(y)


# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#print(y_test)
# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
scaler = StandardScaler()
# Don't cheat - fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)
'''



#X_train = sc.inverse_transform(X_train) # transform back
#X_test = sc.inverse_transform(X_test)

#print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
# Fitting MLP to the Training set

from sklearn.neural_network import MLPClassifier
#classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes= (5, 2), random_state=1, max_iter=100, warm_start=True)

classifier.fit(X_train, y_train)
#classifier.fit(X, y)



# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#print(cm)

'''
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

#print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-1", X1)
#print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-2", X2)
#print(X2.ravel())

#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))



plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = labelencoder_y.inverse_transform(j))
plt.title('MLP (Training set)')
plt.xlabel('Time_Limit (m)')
plt.ylabel('Required_Cpus ')
plt.legend()
plt.show()

'''


'''
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('MLP (Test set)')
plt.xlabel('Time_Limit (m)')
plt.ylabel('Required_Cpus')
plt.legend()
plt.show()

'''
#***********************************

# Dummy classifier is used for getting baseline accuracy
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy='stratified', random_state=None, constant=None)
baseline.fit(X_train, y_train)
ybaselinepred = baseline.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print("Baseline accuracy Accuracy:%.3f%%" % (accuracy_score(y_test, ybaselinepred) * 100.0))
print(classification_report(y_test, ybaselinepred))


# Calculate Accuracy Rate by using accuracy_score()

print("Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, y_pred))