# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Importing the dataset

'''
dataset = pd.read_csv('statistics8.csv')
X = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, 3].values
'''

#file = pd.read_csv('test.csv', sep=',', header= 0)
file = pd.read_csv('flnClass.csv', sep=',')

le = preprocessing.LabelEncoder()
file = file.apply(le.fit_transform)


df = pd.DataFrame(file)

#y = df.state
#X = df.drop('state', axis=1)
#X = df.iloc[:, [0, 1]].values
#y = df.iloc[:, 3].values

y = file.state
X = file.drop('state', axis=1)


#X = df.iloc[:, [0, 12]].values
#y = df.iloc[:, -1].values





'''
dataset = pd.read_csv('statistics8.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 3].values
'''

'''
file = pd.read_csv('statistics8.csv', sep=',')
dataset = pd.DataFrame(file)
y = dataset.state
X = dataset.drop('state', axis=1).drop('task_time', axis=1)
#X = dataset.drop('state', axis=1)

#print('xxxxxxxxxxxxxxxxxxxxxxxx', X)
'''

# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#print('yyyyyyyyyyyyyyyyyyyyyyyyyyyy', y)


# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))



plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = labelencoder_y.inverse_transform(j))
plt.title('Naive Bayes (Training set)')
plt.xlabel('Time_Limit (m)')
plt.ylabel('Required_Cpus ')
plt.legend()
plt.show()

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
plt.title('Naive Bayes (Test set)')
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
