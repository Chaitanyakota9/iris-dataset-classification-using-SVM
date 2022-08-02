#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
# Load the data
df = pd.read_csv('iris.data', names=columns)
df.head()


# In[7]:


df.describe()


# In[8]:


sns.pairplot(df,hue='Class_labels')


# In[9]:


data = df.values


# In[10]:


data


# In[11]:


x = data[:,0:4]


# In[12]:


x


# In[13]:


y = data[:,4]


# In[14]:


y


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30)


# In[19]:


from sklearn.svm import SVC


# In[20]:


model = SVC(gamma = 'scale')
model.fit(x_train,y_train)


# In[21]:


prediction = model.predict(x_test)


# In[23]:


from sklearn.metrics import accuracy_score


# In[33]:


acc = round(accuracy_score(y_test,prediction),2)*100


# In[34]:


acc

