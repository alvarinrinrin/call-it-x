import glob
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

#Generates the dataSet. Images must be in a folder named "images"
#Store a file with the dataSet named "data_set.npz"
def generateDataSet():
	fileNameList = glob.glob('./images/n*')
	dataSet = np.array(len(fileNameList)*[257*[0.]])
	print(dataSet.shape)
	[dataSet,naIndex] = updateDataSet(dataSet,'./images/na_*',0,0)
	[dataSet,nbIndex] = updateDataSet(dataSet,'./images/nb_*',naIndex,1)
	[dataSet,ncIndex] = updateDataSet(dataSet,'./images/nc_*',nbIndex,2)
	[dataSet,ndIndex] = updateDataSet(dataSet,'./images/nd_*',ncIndex,3)
	[dataSet,neIndex] = updateDataSet(dataSet,'./images/ne_*',ndIndex,4)
	np.savez('data_set',dataSet=dataSet,naLength=naIndex,nbLength=nbIndex-naIndex,ncLength=ncIndex-nbIndex,ndLength=ndIndex-ncIndex,neLength=neIndex-ndIndex)

def updateDataSet(dataSet,namePattern,begin,classID):
	fileNameList = glob.glob(namePattern)
	i = begin
	for name in fileNameList:
		img = imread(name,True)
		[h,foo]=np.histogram(img.flatten(),range(256))
		dataSet[i][256] = classID
		[dataSet,i] = copyHist(dataSet,h,i)
		print(name)
	return [dataSet,i]

def copyHist(dataSet,hist,index):
	theSum=sum(hist)
	for i in range(len(hist)):
		dataSet[index][i] = hist[i]/float(theSum)
	return [dataSet,index+1]

#Generates the prototipe samples
def generateClassesMean():
	[dataSet,naLength,nbLength,ncLength,ndLength,neLength]=loadDataSet()
	meanSet = np.array(5*[257*[0.]])
	meanSet = updateMeanSet(meanSet,0,dataSet,0,naLength)
	index = naLength
	meanSet = updateMeanSet(meanSet,1,dataSet,index,index+nbLength)
	index = index + nbLength
	meanSet = updateMeanSet(meanSet,2,dataSet,index,index+ncLength)
	index = index + ncLength
	meanSet = updateMeanSet(meanSet,3,dataSet,index,index+ndLength)
	index = index + ndLength
	meanSet = updateMeanSet(meanSet,4,dataSet,index,index+neLength)
	return meanSet

def loadDataSet():
	dataSet=np.load('./data_set.npz_FILES/dataSet.npy')
	naLength=np.load('./data_set.npz_FILES/naLength.npy')
	nbLength=np.load('./data_set.npz_FILES/naLength.npy')
	ncLength=np.load('./data_set.npz_FILES/naLength.npy')
	ndLength=np.load('./data_set.npz_FILES/naLength.npy')
	neLength=np.load('./data_set.npz_FILES/naLength.npy')
	return [dataSet,naLength,nbLength,ncLength,ndLength,neLength]

def updateMeanSet(meanSet,index,dataSet,begin,end):
	for i in range(len(meanSet[0][:])):
		meanSet[index][i] = np.mean(dataSet[begin:end,i])
	return meanSet

#Classifies an image given a meanSet obtained previously
def classify(meanSet,img):
	[h,foo] = np.histogram(img.flatten(),range(256))
	h = h/float(sum(h))
	distances = 5*[0.]
	for i in range(5):
		distances[i] = np.linalg.norm(h-meanSet[i,0:255])
	return np.argmin(distances)
	
