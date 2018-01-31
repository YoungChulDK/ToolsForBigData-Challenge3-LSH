#Import libraries
import pylab
import imageio
import numpy as np
import time
import imagehash
from skimage import color
from skimage import exposure
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from matplotlib import pyplot as plt

#Get image mean function
def getImgMean(filename):
	#Start time
	t1 = time.time()
	
	#Read videofile with ffmpeg
	vid = imageio.get_reader(filename,  'ffmpeg')
	#Set frames to skip each iteration
	skip = 4
	#Frames after skip
	N = int(vid.get_length()/skip)
	#Get video resolution
	w,h = vid.get_meta_data()['size']
	#print(w,h)
	
	#Initialize mean array
	avg = np.zeros((h,w),np.float)
	#print(type(vid))	
	
	#Find mean image
	for i,im in enumerate(vid):
		if i%skip==0:
			im = color.rgb2gray(im)
			im = exposure.equalize_hist(im)
			im = np.array(im*128,dtype=np.float)
			avg = avg+im/N

	avg=np.array(np.round(avg),dtype=np.uint8)
	#Time elapsed	
	elapsedTime = time.time()-t1
	return avg, elapsedTime

def create_vector(str):
	vec = []
	for c in str:
		vec.append(float(ord(c)))
	return vec
##------------------Main--------------------##

#4 videos, from 2 categories

filename1 = '1C4R01VC8HVS.mp4'
filename2 = '0ACLYF0HM2KX.mp4'

filename3 = '0A6HPR9NHFMF.mp4'
filename4 = '0OBAQSL3HY6B.mp4'

#Get image mean of each video and elapsed time
imgMean, elapsedTime = getImgMean(filename1)
imgMean2, elapsedTime2 = getImgMean(filename2)
imgMean3, elapsedTime3 = getImgMean(filename3)
imgMean4, elapsedTime4 = getImgMean(filename4)

#From array form to PIL Image
pilMean = Image.fromarray(imgMean)
pilMean2 = Image.fromarray(imgMean2)
pilMean3 = Image.fromarray(imgMean3)
pilMean4 = Image.fromarray(imgMean4)

#Show a mean image
#pylab.imshow(imgMean)
#pylab.show()

#Get mean image hash
imgHash = imagehash.average_hash(pilMean)
imgHashP = imagehash.phash(pilMean)

imgHash2 = imagehash.average_hash(pilMean2)
imgHash2P = imagehash.phash(pilMean2)

imgHash3 = imagehash.average_hash(pilMean3)
imgHash3P = imagehash.phash(pilMean3)

imgHash4 = imagehash.average_hash(pilMean4)
imgHash4P = imagehash.phash(pilMean4)

##---------------KMeans-----------------##

print(create_vector(str(imgHash)))

X = np.array([create_vector(str(imgHash)),create_vector(str(imgHashP)), create_vector(str(imgHash2)), create_vector(str(imgHash2P)), create_vector(str(imgHash3)), create_vector(str(imgHash3P)), create_vector(str(imgHash4)), create_vector(str(imgHash4P))])

Nclusters = 4

centroids, cls, inertia = k_means(X, Nclusters, verbose=True, max_iter=100, n_init=1)
print(X.shape)
for c in cls:
	print(c)
