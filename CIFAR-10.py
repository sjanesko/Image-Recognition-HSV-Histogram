from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import scipy
from cv2.cv2 import calcHist
import sys
from scipy.spatial import distance
import pickle
import random

# print entire np array 
np.set_printoptions(threshold=sys.maxsize)

# load keras data
#(trainX, trainXy), (testX, testY) = cifar10.load_data()

# load batches
def unpickle(batch):
    with open('cifar-10-batches/data_batch_' + str(batch), 'rb') as file:
        batch = pickle.load(file, encoding='latin1')

    images = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return images, labels

def loadLabelNames():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def hsvHist(image):
    # convert to HSV
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # normalize
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    # k-means
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 24
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # reshape
    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((img.shape))
 
    # seperate h, s, v channels
    chanH, chanS, chanV = cv2.split(img)

    # flatten arrays 
    chanH = chanH.flatten()
    chanS = chanS.flatten()
    chanV = chanV.flatten()

    # histogram for each channel
    chanH = cv2.calcHist([chanH], [0], None, [8], [0, 180]).flatten()
    chanS = cv2.calcHist([chanS], [0], None, [4], [0, 256]).flatten()
    chanV = cv2.calcHist([chanV], [0], None, [2], [0, 256]).flatten()

    # commented out so every iteration doesnt show the HSV plot
    # # plot
    # plt.subplot(1, 3, 1)
    # plt.title("Hue")
    # plt.bar([1, 2, 3, 4, 5, 6, 7, 8], chanH)


    # plt.subplot(1, 3, 2)
    # plt.title("Saturation")
    # plt.bar([1, 2, 3, 4], chanS)

    # plt.subplot(1, 3, 3)
    # plt.title("Value")
    # plt.bar([1, 2], chanV)

    # plt.show()
    
    # return flattened arr
    return np.hstack([chanH, chanS, chanV])

def getHSVDistance(img1idx, img2idx, t):
    d = distance.euclidean(t[img1idx], t[img2idx])
    return d

# load data
images, labels = unpickle(2)
labelNames = loadLabelNames()

# truncate data so sample size for each class is greater than 100 and less than 150
images = images[:1200]
labels = labels[:1200]

# make easier structure to work with "(image, label)"
data = list(zip(images, labels))

# take the 100 random elements
random.shuffle(data)
data = data[:100]

histArr = [0] * 100

# compute histogram for every image
for i, image in enumerate(data):
    histArr[i] = hsvHist(image[0])

# create distance arr "(distance, index)"
distanceArr = [[0 for x in range(100)] for y in range(100)] 

for i, image1 in enumerate(data):
    for j, image2 in enumerate(data):
        distanceArr[i][j] = (getHSVDistance(i, j, histArr), j, image2[1])

# sort distance array
for i, arr in enumerate(distanceArr):
    arr.sort(key = lambda x: x[0])

print(distanceArr[0][0:11])

plt.imshow(data[0][0])
plt.title(labelNames[data[0][1]])
plt.show()

'''DOESNT WORK RIGHT NOW'''
# fig = plt.figure()
# a = fig.add_subplot(10, 1)

# show similiar images
for i, x in enumerate(distanceArr[0][1:11]):
    plt.imshow(data[x[1]][0])
    plt.title(labelNames[x[2]])
    plt.show()