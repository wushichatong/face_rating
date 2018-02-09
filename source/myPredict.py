
# coding: utf-8

# In[7]:


from sklearn.externals import joblib
import numpy as np
from sklearn import decomposition


# In[8]:


clf = joblib.load(r'model\my_face_rating.pkl')
features = np.loadtxt(r'data\features_All.txt', delimiter = ',')
my_features = np.loadtxt(r'data\my_features.txt', delimiter = ',')
pca = decomposition.PCA(n_components=20)
pca.fit(features)


# In[9]:


predictions = np.zeros([6,1])


# In[10]:


for i in range(0, 6):
    features_test = features[i, :]
    features_test = pca.transform(features_test)
    preditions[i] = clf.predict(features_test)

