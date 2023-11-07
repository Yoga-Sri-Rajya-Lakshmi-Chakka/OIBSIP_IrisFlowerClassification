#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


iris = pd.read_csv("E:/IRIS.csv")


# In[4]:


iris.head()


# In[5]:


iris.tail()


# In[6]:


iris.describe()


# In[7]:


iris.isnull().sum()


# In[8]:


iris.shape


# In[9]:


iris["species"].unique()


# In[10]:


import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()


# In[11]:


x = iris.drop("species", axis=1)
y = iris["species"]


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)


# In[14]:


x_new = np.array([[5, 2.9, 1, 0.2]])


# In[15]:


prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[ ]:




