# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:31:13 2019

@author: Anshum
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from copy import deepcopy

data=pd.read_csv("Ecommerce Customers.csv")

dataset=data.iloc[:,3:]
X=dataset.loc[:,['Avg. Session Length','Time on App','Length of Membership']]
Y=dataset.iloc[:,-1:]
corr_matrix_dataset=dataset.corr()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressor,X=X_train,y=Y_train,cv=10)

def r_squared(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r^2 = {:.3f}".format(r**2),
                xy=(.1, .9), xycoords=ax.transAxes)
    
sns.set(style="ticks", color_codes=True)
linearity_assumption_plot=sns.pairplot(X,kind="reg")
linearity_assumption_plot.map_lower(r_squared)

error_residual=pd.DataFrame(Y_test-Y_pred)

linearity_test_df=pd.DataFrame()
linearity_test_df=deepcopy(X_test)
linearity_test_df['Residuals']=error_residual['Yearly Amount Spent']

linearity_assumption_plot_2=sns.pairplot(linearity_test_df,kind="reg")
linearity_assumption_plot_2.map_lower(r_squared)

endoginity=linearity_test_df.corr()

residual_test=np.column_stack([Y_test,Y_pred])
residual_test=pd.DataFrame(residual_test)
residual_test.columns="Y_test Predictions".split()
sns.jointplot(x="Y_test",y="Predictions",data=residual_test,kind="reg")
stats.levene(residual_test['Y_test'],residual_test['Predictions'])

stats.shapiro(error_residual['Yearly Amount Spent'])

x=sm.add_constant(X)

regressor_OLS=sm.OLS(endog=Y,exog=x).fit()
regressor_OLS.summary()

'''
x_opt=x.loc[:,['const','Avg. Session Length','Time on App','Length of Membership']]
regressor_OLS=sm.OLS(endog=Y,exog=x_opt).fit()
regressor_OLS.summary()
'''
