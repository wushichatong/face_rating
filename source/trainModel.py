
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


# In[12]:


features = np.loadtxt(r'data\features_All.txt', delimiter = ',')


# In[13]:


features_train = features[:-50]
features_test = features[-50:]


# In[14]:


pca = decomposition.PCA(n_components = 20)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)


# In[15]:


ratings = np.loadtxt(r'data\ratings.txt', delimiter = ',')
ratings_train = ratings[0:-50]
rating_test = ratings[-50:]


# In[21]:


regr = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=1.0, random_state=0)
regr = regr.fit(features_train, ratings_train)
joblib.dump(regr, r'model\my_face_rating.pkl', compress=1)
print('Generate Model Successfully')

