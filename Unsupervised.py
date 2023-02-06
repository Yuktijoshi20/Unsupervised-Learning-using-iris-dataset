#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x=[4,5,10,4,10,8,3,2,7,3]
y=[23,27,22,30,21,23,27,29,30,25]
data=list(zip(x,y))
data


# In[5]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
model.fit(data)


# In[6]:


plt.scatter(x,y,c=model.labels_)
plt.show()


# In[10]:



p=[9,3,4,5,2,4,44,4,4,55]
r=[3,4,5,22,55,22,44,88,77,66]
data=list(zip(r,p))
data


# In[11]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters=7)
model.fit(data)


# In[9]:


plt.scatter(r,p,c=model.labels_)
plt.show()


# In[15]:


import pandas as pd
df1=pd.read_csv("C:/Users/DITU.DESKTOP-KNV4SBH/Downloads/1/Iris.csv")
df1


# In[22]:


X=df1["PetalLengthCm"]
Y=df1["SepalWidthCm"]
data=list(zip(X,Y))
data

from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
model.fit(data)
plt.scatter(X,Y,c=model.labels_)
plt.show()


# In[ ]:




