import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle 
from scipy.spatial import distance
import random

labelNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# load batches
def unpickle(batch):
    with open('cifar-10-batches/data_batch_' + str(batch), 'rb') as file:
        batch = pickle.load(file, encoding='latin1')

    images = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return images, labels

class imageObj:
    hsvHistogram = None
    kmeansModel = None
    def __init__(self, image,  label, sampleSize):
        self.img = image
        self.label = label
        self.nearestNeighbors = [[0 for x in range(sampleSize)] for y in range(sampleSize)] 

    def showImage(self):
        plt.imshow(self.img)
        plt.title(labelNames[self.label])
        plt.show()

    def calcHistogram(self):
        # convert to HSV
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

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

        self.hsvHistogram = np.hstack([chanH, chanS, chanV])

    def showHist(self):
        # plot
        plt.subplot(1, 3, 1)
        plt.title("Hue")
        plt.bar([1, 2, 3, 4, 5, 6, 7, 8], self.hsvHistogram[:8])

        plt.subplot(1, 3, 2)
        plt.title("Saturation")
        plt.bar([1, 2, 3, 4], self.hsvHistogram[8:12])

        plt.subplot(1, 3, 3)
        plt.title("Value")
        plt.bar([1, 2], self.hsvHistogram[12:])

        plt.show()

    def calcKmeans(self, K = 64):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # normalize
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

        # k-means
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # reshape
        center = np.uint8(center)
        res = center[label.flatten()]
        img = res.reshape((img.shape))

        self.kmeansModel = img

    def showKmeansModel(self):
        plt.imshow(self.kmeansModel)
        plt.show()

    def getHSVDistance(self, otherImage):
        return distance.euclidean(self.kmeansModel.flatten(), otherImage.kmeansModel.flatten())

    def calcNearestNeighbors(self, imageArr):
        for index, imageObj in enumerate(imageArr):
            self.nearestNeighbors[index] = (self.getHSVDistance(imageObj), imageObj)
        self.nearestNeighbors.sort(key = lambda x: x[0])
        
    def getNearestNeighbors(self):
        return self.nearestNeighbors[1:11]

images, labels = unpickle(2)
sampleSize = 100

# truncate data so sample size for each class is greater than 100 and less than 150
images = images[:5000]
labels = labels[:5000]

# make easier structure to work with "(image, label)"
data = list(zip(images, labels))

# take sampleSize random elements
random.shuffle(data)
data = data[:sampleSize]

# create array to store imageObjs
imageObjArr = []

# create object
for i in range(sampleSize):
    imageObjArr.append(imageObj(data[i][0], data[i][1], sampleSize))
    # imageObjArr[i].showImage()
    imageObjArr[i].calcKmeans(16)
    # imageObjArr[i].showKmeansModel()

for i in imageObjArr:
    i.calcNearestNeighbors(imageObjArr)

imageObjArr[0].showImage()

#show similar images
for image in imageObjArr[0].getNearestNeighbors():
    plt.imshow(image[1].img)
    plt.title("Label: " + labelNames[image[1].label] + '\n' + "Distance: " + str(image[0]))
    plt.show()
