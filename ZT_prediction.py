#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv(r"C:\Users\pcd\Desktop\ZT_filtered.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.corr()


# In[8]:


sns.heatmap(df.corr(),annot=True)


# In[9]:


sns.pairplot(df,kind='kde')


# In[10]:


sns.boxplot(df['seebeck_coefficient'])


# In[11]:


sns.boxplot(df['power_factor'])


# In[12]:


sns.boxplot(df['thermal_conductivity'])


# In[13]:


sns.boxplot(df['electrical_conductivity'])


# In[14]:


df.hist(bins=5,figsize=(13,13))


# In[ ]:





# In[15]:


#splitting
x=df.iloc[:,:6].values
y=df.iloc[:,6].values


# In[16]:


le=LabelEncoder()
x[:,0]=le.fit_transform(x[:,0])


# In[17]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[18]:


lr=LinearRegression()
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)
lr_acc=r2_score(y_test,lr_pred)
lr_acc


# In[19]:


lr_rmse=np.sqrt(mean_squared_error(lr_pred,y_test))
print(lr_rmse)
lr_mae=mean_absolute_error(y_test,lr_pred)
print(lr_mae)


# In[20]:


rf=RandomForestRegressor(1200)
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_test)
rf_acc=r2_score(y_test,rf_pred)
rf_acc


# In[38]:


rf_rmse=np.sqrt(mean_squared_error(rf_pred,y_test))
print(rf_rmse)
rf_mae=mean_absolute_error(y_test,rf_pred)
print(rf_mae)


# In[ ]:





# In[22]:


import xgboost as xgb
xg= xgb.XGBRegressor(max_depth = 5, n_estimators = 1000)
xg.fit(x_train,y_train)
xg_pred=xg.predict(x_test)
xg_acc=r2_score(y_test,xg_pred)
xg_acc


# In[40]:


xg_rmse=np.sqrt(mean_squared_error(xg_pred,y_test))
print(xg_rmse)
xg_mae=mean_absolute_error(y_test,xg_pred)
print(xg_mae)


# In[23]:


from sklearn.ensemble import BaggingRegressor
bg= BaggingRegressor(base_estimator=RandomForestRegressor(1200))
bg.fit(x_train,y_train)
bg_pred=bg.predict(x_test)
bg_acc=r2_score(y_test,bg_pred)
bg_acc


# In[41]:


bg_rmse=np.sqrt(mean_squared_error(bg_pred,y_test))
print(bg_rmse)
bg_mae=mean_absolute_error(y_test,bg_pred)
print(bg_mae)


# In[24]:


from sklearn.ensemble import GradientBoostingRegressor
gbr= GradientBoostingRegressor()
gbr.fit(x_train,y_train)
gbr_pred=gbr.predict(x_test)
gbr_acc=r2_score(y_test,gbr_pred)
gbr_acc


# In[42]:


gbr_rmse=np.sqrt(mean_squared_error(gbr_pred,y_test))
print(gbr_rmse)
gbr_mae=mean_absolute_error(y_test,gbr_pred)
print(gbr_mae)


# In[43]:


gbr_p={
 'n_estimators':[ x for x in range(1000,2000,200)],
    'criterion':['friedman_mse','squared_error','mse'], 
    
    
}


grid_gbr=GridSearchCV(GradientBoostingRegressor(),gbr_p,n_jobs=-1,cv=5,verbose=True,pre_dispatch='2 *n_jobs')
grid_gbr.fit(x_train,y_train)
pred_gbr=grid_gbr.predict(x_test)
gbr=r2_score(y_test,pred_gbr)
print('Accuracy of GBM model:',gbr)


# In[44]:


grid_gbr.best_estimator_


# In[54]:


gbr= GradientBoostingRegressor(n_estimators=1200)
gbr.fit(x_train,y_train)
gbr_pred=gbr.predict(x_test)
gbr_acc=r2_score(y_test,gbr_pred)
gbr_acc


# In[56]:


gbr_rmse=np.sqrt(mean_squared_error(gbr_pred,y_test))
print(gbr_rmse)
gbr_mae=mean_absolute_error(y_test,gbr_pred)
print(gbr_mae)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




