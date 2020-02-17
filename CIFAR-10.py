from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import scipy
from cv2.cv2 import calcHist
import sys
from scipy.spatial import distance
import pickle

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

    # plot
    plt.subplot(1, 3, 1)
    plt.title("Hue")
    plt.bar([1, 2, 3, 4, 5, 6, 7, 8], chanH)


    plt.subplot(1, 3, 2)
    plt.title("Saturation")
    plt.bar([1, 2, 3, 4], chanS)

    plt.subplot(1, 3, 3)
    plt.title("Value")
    plt.bar([1, 2], chanV)

    plt.show()
    
    # return flattened arr
    return np.hstack([chanH, chanS, chanV])

images, labels = unpickle(2)

# truncate data so sample size for each class is > 100 and < 150
images = images[:1200]
labels = labels[:1200]

for i in range (10):
    print(labels.count(i))