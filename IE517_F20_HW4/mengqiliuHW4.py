#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:59:31 2020

@author: liumengqi
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from mlxtend.plotting import heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv('https://raw.githubusercontent.com/rasbt/'
               'python-machine-learning-book-2nd-edition'
               '/master/code/ch10/housing.data.txt',
               header=None,
               sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols],size=2.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()


boxplot = df.boxplot(column=['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
plt.show()


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)




X = df[['RM']].values
y = df['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)



sns.reset_orig() 
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()




def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return None




lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()



num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
print("Price in $1000s: %.3f" % sc_y.inverse_transform(price_std))


print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])

slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)




lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()

Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print('Slope: %.3f' % w[1])
print('Intercept: %.3f' % w[0])

ransac = RANSACRegressor(LinearRegression(), 
                         max_trials=100, 
                         min_samples=50, 
                         loss='absolute_loss', 
                         residual_threshold=5.0, 
                         random_state=0)


ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white', 
            marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white', 
            marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='black', lw=2)   
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')

plt.show()


print (slr.coef_)
print('Intercept: %.3f' % ransac.estimator_.intercept_)



X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


#Linear regression
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
ary = np.array(range(100000))



plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()
plt.show()

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))



# Using regularized methods for regression

# Ridge regression:
alpha_space_ridge=np.array([0,0.1,0.5,1])
coef_ridge=[]
y_inter_ridge=[]
mse_train_ridge=[]
mse_test_ridge=[] 
r2_train_ridge=[] 
r2_test_ridge=[]

for i in alpha_space_ridge:
    ridge= Ridge(alpha=i,normalize=True)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    coef_ridge.append(ridge.coef_)
    y_inter_ridge.append(ridge.intercept_)
    y_train_ridge_pred=ridge.predict(X_train) 
    y_test_ridge_pred=ridge.predict(X_test) 
    mse_train_ridge.append(mean_squared_error(y_train,y_train_ridge_pred)) 
    mse_test_ridge.append(mean_squared_error(y_test,y_test_ridge_pred)) 
    r2_train_ridge.append(ridge.score(X_train, y_train)) 
    r2_test_ridge.append(ridge.score(X_test, y_test))


print(slr.coef_)
print('y-intercept for all alpha: {}'.format(y_inter_ridge))


index_ridge = mse_test_ridge.index(min(mse_test_ridge)) 
best_alpha_ridge = alpha_space_ridge[index_ridge] 
print('The best alpha: ', best_alpha_ridge) 
print('coefficient for best alpha: ', coef_ridge[index_ridge]) 
print('y-intercept for best alpha: ', y_inter_ridge[index_ridge])

ridge_best = Lasso(alpha = alpha_space_ridge[index_ridge]) 
ridge_best.fit(X_train, y_train)
y_train_best_ridge_pred = ridge_best.predict(X_train)
y_test_best_ridge_pred = ridge_best.predict(X_test)

plt.scatter(y_train_best_ridge_pred, y_train_best_ridge_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_best_ridge_pred, y_test_best_ridge_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()


# LASSO regression:
lasso = Lasso()

alpha_space=np.array([0,0.1,0.5,1])
coef_lasso=[]
y_inter_lasso=[]
mse_train_lasso=[]
mse_test_lasso=[] 
r2_train_lasso=[] 
r2_test_lasso=[]

for j in alpha_space:
    lasso=Lasso(alpha=j,normalize=True)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    coef_lasso.append(lasso.coef_)
    y_inter_lasso.append(lasso.intercept_)
    mse_train_lasso.append(mean_squared_error(y_train, y_train_pred))
    mse_test_lasso.append(mean_squared_error(y_test, y_test_pred))
    r2_train_lasso.append(lasso.score(X_train, y_train)) 
    r2_test_lasso.append(lasso.score(X_test, y_test))

print('Pick alpha :{}'.format(alpha_space)) 
print('coefficient for all alpha: {}'.format(coef_lasso)) 
print('y-intercept for all alpha: {}'.format(y_inter_lasso))


index_lasso = mse_test_lasso.index(min(mse_test_lasso)) 
best_alpha_lasso = alpha_space[index_lasso] 
print('The best alpha: ', best_alpha_lasso) 
print('coefficient for best alpha: ', coef_lasso[index_lasso]) 
print('y-intercept for best alpha: ', y_inter_lasso[index_lasso])

lasso_best = Lasso(alpha = alpha_space[index_lasso]) 
lasso_best.fit(X_train, y_train)
y_train_best_lasso_pred = lasso_best.predict(X_train)
y_test_best_lasso_pred = lasso_best.predict(X_test)

plt.scatter(y_train_best_lasso_pred, y_train_best_lasso_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_best_lasso_pred, y_test_best_lasso_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()