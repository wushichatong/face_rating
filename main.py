
# coding: utf-8

# In[1]:


import cv2
import dlib
import math
import itertools
import numpy as np
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import os


# In[2]:


def load_data():
    features = np.loadtxt(r'data\features_All.txt', delimiter=',')
    ratings = np.loadtxt(r'data\ratings.txt', delimiter=',')
    return features, ratings


# In[3]:


def train_save_model(features, ratings):
    # # load all features
    # features = np.loadtxt(r'data\features_All.txt', delimiter=',')
    # # seperate datasets into train and test
    # features_train = features[:-50]
    # features_test = features[-50:]
    #
    # # load labels
    # ratings = np.loadtxt(r'data\ratings.txt', delimiter=',')
    # ratings_train = ratings[0:-50]
    # ratings_test = ratings[-50:]
    #
    # # dimensional reducing
    # pca = decomposition.PCA(n_components=20)
    # pca.fit(features_train)
    # features_train = pca.transform(features_train)
    # features_test = pca.transform(features_test)
    #
    # regr = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
    # regr = regr.fit(features_train, ratings_train)
    # joblib.dump(regr, r'model\my_face_rating.pkl', compress=1)
    # print('Generate Model Successfully')
    predictions = np.zeros(ratings.size)

    for i in range(0, 500):
        features_train = np.delete(features, i, 0)
        features_test = features[i, :]
        ratings_train = np.delete(ratings, i, 0)
        ratings_test = ratings[i]
        pca = decomposition.PCA(n_components=20)
        pca.fit(features_train)
        features_train = pca.transform(features_train)
        features_test = pca.transform(features_test.reshape(1, -1))
        regr = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
        regr.fit(features_train, ratings_train)
        predictions[i] = regr.predict(features_test)
#         print('predictions[{}]:{}'.format(i, predictions[i]))
        print('number of models trained:', i + 1)
        
    pca.fit(features)
    features = pca.transform(features)
    # regr.fit(features_train, ratings_train)
    ratings_predict = regr.predict(features)
    corr = np.corrcoef(ratings_predict, ratings)[0, 1]
    print('Correlation:', corr)
    
    truth, = plt.plot(ratings_test, 'r')
    prediction, = plt.plot(ratings_predict, 'b')
    plt.legend([truth, prediction], ["Ground Truth", "Prediction"])

    plt.show()
    joblib.dump(regr, r'model\my_face_rating.pkl', compress=1)
#     return regr


# In[22]:


def get_landmarks(fileName):
    PREDICTOR_PATH = r'data\shape_predictor_68_face_landmarks.dat'
#     fileName = 7
    im = cv2.imread('image/{}'.format(fileName))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    rects = detector(im, 1)
    
    if len(rects) >= 1:
        print('{} faces detected.'.format(len(rects)))
    if len(rects) == 0:
        print('No face detected, please change photoes')
        return 1
    f = open(r'data\landmarks.txt', 'w')
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
        im = im.copy()
        hello = np.array(landmarks.mean(axis = 0))
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            f.write(str(point[0, 0]))
            f.write(',')
            f.write(str(point[0, 1]))
            f.write(',')
            cv2.circle(im, pos, 3, color = (0, 255, 255))
        f.write('\n')
        cv2.putText(im, '{}'.format(i), (int(hello[0][0]),int(hello[0][1])),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),10)
    f.close()
    # print('{}, get!'.format(fileName))
    cv2.imwrite(r'image_with_features\{}'.format(fileName),im, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
    return 1
    


# In[5]:


def facialRatio(points):
    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]
    x3 = points[4]
    y3 = points[5]
    x4 = points[6]
    y4 = points[7]
    
    dist1 = math.sqrt((x1- x2)**2 + (y1 - y2)**2)
    dist2 = math.sqrt((x3- x4)**2 + (y3 - y4)**2)
    ratio = dist1/dist2
    return ratio


# In[11]:


def generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, allLandmarkCoordinates):
    size = allLandmarkCoordinates.shape
    allLandmarkCoordinates = allLandmarkCoordinates.reshape(-1, 136)
#     print(size)
    size = allLandmarkCoordinates.shape
#     print(size)
    allFeatures = np.zeros((size[0], len(pointIndices1)))
    for x in range(size[0]):
        landmarkCoordinates = allLandmarkCoordinates[x, :]
        ratios = [];
        for i in range(0, len(pointIndices1)):
            x1 = landmarkCoordinates[2*(pointIndices1[i]-1)]
            y1 = landmarkCoordinates[2*pointIndices1[i] - 1]
            x2 = landmarkCoordinates[2*(pointIndices2[i]-1)]
            y2 = landmarkCoordinates[2*pointIndices2[i] - 1]

            x3 = landmarkCoordinates[2*(pointIndices3[i]-1)]
            y3 = landmarkCoordinates[2*pointIndices3[i] - 1]
            x4 = landmarkCoordinates[2*(pointIndices4[i]-1)]
            y4 = landmarkCoordinates[2*pointIndices4[i] - 1]

            points = [x1, y1, x2, y2, x3, y3, x4, y4]
            ratios.append(facialRatio(points))
        allFeatures[x, :] = np.asarray(ratios)
        
    return allFeatures


# In[7]:


def generateAllFeatures(allLandmarkCoordinates):
    a = [18, 22, 23, 27, 37, 40, 43, 46, 28, 32, 34, 36, 5, 9, 13, 49, 55, 52, 58]
    combinations = itertools.combinations(a, 4)
    i = 0 
    pointIndices1 = []
    pointIndices2 = []
    pointIndices3 = []
    pointIndices4 = []
    
    for combination in combinations:
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[1])
        pointIndices3.append(combination[2])
        pointIndices4.append(combination[3])
        i = i + 1
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[2])
        pointIndices3.append(combination[1])
        pointIndices4.append(combination[3])
        i = i + 1
        pointIndices1.append(combination[0])
        pointIndices2.append(combination[3])
        pointIndices3.append(combination[1])
        pointIndices4.append(combination[2])
        i = i + 1
    return generateFeatures(pointIndices1, pointIndices2, pointIndices3, pointIndices4, allLandmarkCoordinates)


# In[8]:


def save_features(fileName):
    if(get_landmarks(fileName)):
        landmarks = np.loadtxt(r'data\landmarks.txt', delimiter = ',', usecols = list(range(136)))
        featuresAll = generateAllFeatures(landmarks)
        np.savetxt(r'data/my_features.txt', featuresAll, delimiter = ',', fmt = '%.04f')
        return 1
        print('Generate Feature Successfully')
    else:
        return 0


# In[32]:


fileName = input("Please input filename:")
features, ratings = load_data()
if os.path.exists(r'model\my_face_rating.pkl') == False:
    train_save_model(features, ratings)# Only need to run once when you initiate the program
    
clf = joblib.load(r'model\my_face_rating.pkl')
if save_features(fileName):
    print('Save features successful! ')
    my_features = np.loadtxt(r'data\my_features.txt', delimiter = ',')
    pca = decomposition.PCA(n_components=20)
    pca.fit(features)
    my = my_features.reshape(-1, 11628)
    my = pca.transform(my)
    predictions = clf.predict(my)
    print(predictions)
    
    for index, prediction in enumerate(predictions):
        print('Index %d: %.4f'%(index, prediction))
#     clf.predict(my)
else: 
    print(' Save features failed ')

