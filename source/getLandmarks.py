
# coding: utf-8

# In[14]:


import cv2
import dlib
import numpy as np


# In[15]:


PREDICTOR_PATH = r'data\shape_predictor_68_face_landmarks.dat'


# In[16]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# In[17]:


class NoFaces(Exception):
    pass


# In[18]:


im = cv2.imread('image/landmarks.jpg')


# In[19]:


rects = detector(im, 1)


# In[20]:


if len(rects) >= 1:
    print('{} faces detected.'.format(len(rects)))


# In[21]:


if len(rects) == 0:
    raise NoFaces    


# In[22]:


f = open(r'data\landmarks.txt', 'w')


# In[23]:


for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
    im = im.copy()
    
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        
        f.write(str(point[0, 0]))
        f.write(',')
        f.write(str(point[0, 1]))
        f.write(',')
        cv2.circle(im, pos, 3, color = (0, 255, 0))
    f.write('\n')


# In[24]:


print('Landmarks, get!')


# In[25]:


cv2.imwrite('1.png',im, [int( cv2.IMWRITE_JPEG_QUALITY), 95])

