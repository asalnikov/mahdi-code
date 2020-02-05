

#from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, validation_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

                        #********************************* CPU *************************************#

#----------------------------------------------------------------------------------------

# Load & do first stage pre-processing

'''
file = pd.read_csv('statistics7.csv', sep=',')
df = pd.DataFrame(file)
df = df[df.task_time > 0]

df = df[df.required_cpus > 0]
'''

file = pd.read_csv('flnTime.csv', sep=',', header= 0)
#file = pd.read_csv('fstatCPU.csv', sep=',', header= 0)


le = preprocessing.LabelEncoder()
file = file.apply(le.fit_transform)


df = pd.DataFrame(file)

#df = le.fit_transform(df.astype(str))

# Normalise data
#data=df
data = df[np.abs(df["required_cpus"]-df["required_cpus"].mean())<=(6*df["required_cpus"].std())]

data = np.abs((data - data.mean()) / (data.std()))
#data = (data - data.mean()) / (data.std())


#y = data.task_time
y = data.required_cpus
x = data.drop('required_cpus', axis=1)


#x = data.iloc[:, [0, 12]].values
#y = data.iloc[:, -1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

plt.style.use('seaborn-dark-palette')


#----------------------------------------------------------------------------------------







### Ordinary Least Squares (OLS) Regression

lm = LinearRegression()
lm.fit(x_train, y_train)
predicted = lm.predict(x_test)


plt.scatter(y_test, predicted)
plt.plot([-1,6], [-1,6], "r--", lw=1, alpha=0.4)
plt.xlabel("Targeted_Required_CPUs")
plt.ylabel("Predicted_Required_CPUs")
plt.axis([-1,6,-1,6])
plt.text(-0.7,5.5, ' R-squared = {}'.format(round(float(lm.score(x_test,y_test)), 2)))

print("lm.score(x_test,y_test)==============", lm.score(x_test,y_test))
plt.text(-0.7,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted)), 2)))

print("mean_squared_error(y_test, predicted)===========",float(mean_squared_error(y_test, predicted)) )
plt.title('OLS - Predicted_Required_CPUs vs. Targeted_Required_CPUs')
plt.show()

print("Model coefficients: ", list(zip(list(x_test), lm.coef_)))

### 10 folds cross-validation along the previous OLS Regression

lm = LinearRegression()
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(lm, x, y, cv=shuffle)
print("cv_scoressssssssssssssssss", cv_scores)
print("cv_scores.mean()", cv_scores.mean())


#************************************************************************************
#*****************************************************************************
#----------------------------------------------------------------------------------------


### KNN Regression

# GridsearchCV for KNN regression (CV=10)

parameters = {'n_neighbors':[5, 10, 15, 25, 30]}
knn = KNeighborsRegressor()
grid_obj = GridSearchCV(knn, parameters, cv=10, scoring='r2')
grid_obj.fit(x, y)

print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)
print(pd.DataFrame(grid_obj.cv_results_))

# Validation curve over different ranges of N for KNN regression with 10 fold cross-validation

param_range = tuple(list(range(1,15)))
train_scores, test_scores = validation_curve(KNeighborsRegressor(), x, y, param_name="n_neighbors", param_range=param_range,cv=10)
test_scores_mean = np.mean(test_scores, axis=1)

plt.title("Validation Curve with KNN")
plt.xlabel("KNN")
plt.ylabel("Score")
plt.plot(param_range, test_scores_mean, label="Test score",)
plt.legend(loc="best")
plt.show()

# KNN Regression with the optimal n (n_neighbors = 15)

knn = KNeighborsRegressor(n_neighbors=15)
knn.fit(x_train, y_train)
predicted_knn = knn.predict(x_test)

plt.scatter(y_test, predicted_knn)
plt.plot([-1,6], [-1, 6], "r--", lw=1, alpha=0.4)

plt.xlabel("Targeted_Required_CPUs")
plt.ylabel("Predicted_Required_CPUs")
plt.text(-0.7,5.5, ' R-squared = {}'.format(round(float(knn.score(x_test,y_test)), 2)))
plt.text(-0.7,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_knn)), 2)))
plt.title('KNN (15) - Predicted Required_CPUs vs. Targeted_Required_CPUs')
plt.show()

# 10 folds cross-validation along the previous KNN regression

knn = KNeighborsRegressor(n_neighbors=15)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(knn, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#-------------------------------------------------------------------------



### Lasso Regression

# GridSearchCV for Lasso Regression (CV=10)

parameters={'alpha': [0,0.25,1,5,10,15,20,100]}
lasso_reg = Lasso(max_iter=1500)
grid_obj = GridSearchCV(lasso_reg,parameters,cv=10, scoring = 'r2')
grid_obj.fit(x,y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Lasso Regression along optimal alpha (0)

lasso = Lasso(alpha=0)
lasso.fit(x_train, y_train)
predicted_lasso = lasso.predict(x_test)

plt.scatter(y_test, predicted_lasso)
plt.plot([-1,6], [-1,6], "r--", lw=1, alpha=0.4)

plt.xlabel("Targeted_Required_CPUs")
plt.ylabel("Predicted_Required_CPUs")
plt.axis([-1,6,-1,6])
plt.text(-0.7,5.5, ' R-squared = {}'.format(round(float(lasso.score(x_test,y_test)), 2)))
plt.text(-0.7,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_lasso)), 2)))
plt.title('Lasso (Alpha - 0) - Predicted_Required_CPUs vs. Targeted_Required_CPUs')
plt.show()

# 10 folds cross-validation along the previous Lasso regression

lasso = Lasso(alpha=0)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(lasso, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#-------------------------------------------------------------------------



### Ridge Regression

# GridSearchCV - Ridge Regression

#parameters={'alpha': [0, 0.1, 5,10,15,200]}
parameters={'alpha': [0, 0.1, 5,10,15]}

rdg_reg = Ridge()
grid_obj = GridSearchCV(rdg_reg,parameters,cv=10, scoring = 'r2')
grid_obj.fit(x,y)

results = pd.DataFrame(grid_obj.cv_results_)

plt.plot(results['param_alpha'], results['mean_test_score'], 'r-', alpha=0.4)
#plt.plot(results['mean_test_score'],results['param_alpha'],  'r-', alpha=0.4)

plt.text(0,0.667, ' Best-score = {}'.format(round(float(grid_obj.best_score_), 2)))
plt.text(0,0.666, ' Optimal alpha = {}'.format(grid_obj.best_params_))
plt.xlabel("Alpha")
plt.ylabel("Mean test score")
plt.title('GridsearchCV RidgeRegressor (CV=10)')
plt.show()

# RidgeCV Regression with 10 fold cross-validation along alpha values of 0.1, 1 and 10

Ridge_CV = RidgeCV(alphas=(0.1, 1.0, 20.0), cv=10)
Ridge_CV.fit(x_train, y_train)
predicted_Ridge_CV = Ridge_CV.predict(x_test)

plt.scatter(y_test, predicted_Ridge_CV)
plt.plot([-1,6], [-1,6], "r--", lw=1, alpha=0.4)

plt.xlabel("Targeted_Required_CPUs")
plt.ylabel("Predicted_Required_CPUs")

plt.text(-0.7,5.5, ' R-squared = {}'.format(round(float(Ridge_CV.score(x_test,y_test)), 2)))
plt.text(-0.7,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_Ridge_CV)), 2)))
plt.title('Ridge (Alpha = {}) - Predicted_Required_CPUs vs. Targeted_Required_CPUs'.format(Ridge_CV.alpha_))
plt.show()

# 10 folds cross-validation along the previous Ridge regression

ridge = Ridge(alpha=0.1)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(ridge, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#-------------------------------------------------------------------------






### Polynomial regression (degrees = 4)

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_poly, y, random_state = 0)

poly_reg = LinearRegression().fit(x_train1, y_train1)
predicted_poly = poly_reg.predict(x_test1)

plt.scatter(y_test1, predicted_poly)
plt.plot([-1,6], [-1,6], "r--", lw=1, alpha=0.4)
plt.xlabel("Targeted_Required_CPUs")
plt.ylabel("Predicted_Required_CPUs")
plt.axis([-1,6,-1,6])
plt.text(-0.7,5.5, ' R-squared = {}'.format(round(float(poly_reg.score(x_test1,y_test1)), 2)))
plt.text(-0.7,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test1, predicted_poly)), 2)))
plt.title('Poly (Four degrees) - Predicted_Required_CPUs vs. Targeted_Required_CPUs')
plt.show()

# 10 folds cross-validation along the previous Polynomial regression (degrees =4)

LR = LinearRegression()
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(LR, x_poly, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#------------------------------------------------------------------------





### Support Vector Regression

# GridSearchCV - Support Vector Regression

Cs = [10, 100]
gammas = [0.001, 0.01, 0.1, 1]
parameters = {'C': Cs, 'gamma' : gammas}
grid_obj = GridSearchCV(SVR(kernel='linear'), parameters, cv=3)
grid_obj.fit(x, y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Support Vector Regression along optimal parameters

svr = SVR(kernel='linear', gamma = 0.001, C= 10)

svr.fit(x_train, y_train)
predicted_svr = svr.predict(x_test)

plt.scatter(y_test, predicted_svr)
plt.plot([-1,6], [-1,6], "r--", lw=1, alpha=0.4)

plt.xlabel("Targeted_Required_CPUs")
plt.ylabel("Predicted_Required_CPUs")
plt.text(-0.7,5.5, ' R-squared = {}'.format(round(float(svr.score(x_test,y_test)), 2)))
plt.text(-0.7,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_svr)), 2)))
plt.title('SVR (Gamma = 0.001, C=10) - Predicted_Required_CPUs vs. Targeted_Required_CPUs')
plt.show()

# 10 folds cross-validation along the previous SVR

svr = SVR(kernel='linear', gamma = 0.001, C= 10)

shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(svr, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#------------------------------------------------------------------------------




### Decision Tree Regression

# GridSearchCV - Decision Tree Regression

tree = DecisionTreeRegressor()
parameters = {"max_depth": range(2,5), "random_state":[0], "min_samples_leaf": [6,7,8,9]}
grid_obj = GridSearchCV(estimator=tree,param_grid=parameters, cv=2, scoring='r2')
grid_fit =grid_obj.fit(x,y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Decision Tree Regression along optimal parameters

tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=6)
tree.fit(x_train, y_train)
predicted_tree = tree.predict(x_test)

plt.scatter(y_test, predicted_tree)
plt.plot([-1, 6], [-1,6], "r--", lw=1, alpha=0.4)

plt.xlabel("Targeted_Required_CPUs")
plt.ylabel("Predicted_Required_CPUs")
plt.axis([-0.5,6,-0.5,6])
plt.text(0,5.5, ' R-squared = {}'.format(round(float(tree.score(x_test,y_test)), 2)))
plt.text(0,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_tree)), 2)))
plt.title('Decision Tree Regressor (max_depth =4, min_sample_leaf=8) - Predicted__Required_CPUs vs. Targeted__Required_CPUs')
plt.show()

print("Feature importances: ", list(zip(list(x_test), tree.feature_importances_)))

# 10 folds cross-validation along the previous Decision Tree

tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=6)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(tree, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#------------------------------------------------------------------------------



### RandomForest Regression

# GridSearchCV - Random Forest Regression

forest = RandomForestRegressor()
#parameters = {"n_estimators": [10, 20], "max_features": [4], 'max_depth': [None, 1, 2, 3]}
parameters = {"n_estimators": [10, 20], 'max_depth': [None, 1, 2, 3]}

grid_obj = GridSearchCV(estimator=forest, param_grid=parameters, cv=5, scoring='r2')
grid_obj.fit(x,y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Random Forest regression along optimal parameters

#forest = RandomForestRegressor(n_estimators=20, max_features=4)
forest = RandomForestRegressor(n_estimators=20)

forest.fit(x_train, y_train)
predicted_forest = forest.predict(x_test)

plt.scatter(y_test, predicted_forest)
plt.plot([-1, 6], [-1,6], "r--", lw=1, alpha=0.4)
plt.xlabel("Targeted__Required_CPUs")
plt.ylabel("Predicted__Required_CPUs")
plt.axis([-1,6,-1,6])
plt.text(-0.7,5.5, ' R-squared = {}'.format(round(float(forest.score(x_test,y_test)), 2)))
plt.text(-0.7,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_forest)), 2)))
plt.title('Random Forest Regression (N=20,max_feat=4) - Predicted__Required_CPUs vs. Targeted__Required_CPUs')
plt.show()

# 10 folds cross-validation along the previous Random Forest Regression

#forest= RandomForestRegressor(n_estimators=20, max_features=4)
forest= RandomForestRegressor(n_estimators=20)

shuffle = KFold(n_splits=3, shuffle=True, random_state=0)
cv_scores = cross_val_score(forest, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#------------------------------------------------------------------------------


### Neural Network

### GridSearchCV - Neural Network MLP Regression

parameters = {'alpha':[5, 10, 100]}
MLP = MLPRegressor(hidden_layer_sizes= [3,3], random_state=0, solver='lbfgs')
grid_obj = GridSearchCV(MLP, parameters, cv=5, scoring='r2')
grid_obj.fit(x, y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Neural Network MLP regression along optimal parameters

neural = MLPRegressor(hidden_layer_sizes= [3,3], alpha=5, random_state=0, solver='lbfgs')
neural.fit(x_train, y_train)
predicted_neural = neural.predict(x_test)

plt.scatter(y_test, predicted_neural)
plt.plot([-1, 6], [-1,6], "r--", lw=1, alpha=0.4)

plt.xlabel("Targeted_Required_CPUs")
plt.ylabel("Predicted_Required_CPUs")
plt.axis([-1,6,-1,6])
plt.text(-0.7,5.5, ' R-squared = {}'.format(round(float(neural.score(x_test,y_test)), 2)))
plt.text(-0.7,5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_neural)), 2)))
plt.title('Neural Network MLP Regression (layer=[3.3], alpha=5) - Predicted_Required_CPUs vs. Targeted_Required_CPUs')
plt.show()

# 10 folds cross-validation along the previous Neural Network MLP

MLP = MLPRegressor(hidden_layer_sizes= [3,3], alpha=5, random_state=0, solver='lbfgs')
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)

cv_scores = cross_val_score(MLP, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())


'''
