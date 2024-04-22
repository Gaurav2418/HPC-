#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('C:/Users/shreeyash/Downloads/Boston.csv')
df.head(10)


# In[5]:


df.shape


# In[6]:





# In[7]:


df


# In[8]:


df.head


# In[9]:


df.isnull().sum()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.corr()['MEDV'].sort_values()


# In[13]:


X = df.loc[:,['LSTAT','PTRATIO','RM']]
Y = df.loc[:,"MEDV"]
X.shape,Y.shape


# In[15]:


boston = pd.read_csv('C:/Users/shreeyash/Downloads/Boston.csv')


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k, data=df, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[21]:


sns.boxplot(df.CRIM)


# In[22]:


#checking Correlation of the data 
correlation = df.corr()
correlation.loc['CRIM']


# In[25]:


# Checking the scatter plot with the most correlated features
plt.figure(figsize = (20,5))
features = ['LSTAT','RM','PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = df[col]
    y = df.CRIM
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('"House prices in $1000"')


# In[ ]:




