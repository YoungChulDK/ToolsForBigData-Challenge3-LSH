import os, PIL,time
import numpy as np
import cv2
from PIL import Image


def getImageHash(img):
	(x,y) = (9,8) #Resized image size, 9x8
	imgSmall = img.resize((x,y)) #Resize image
	imgMat = np.asarray(imgSmall) #Convert image to pixel value matrix

	#Compare adjacent pixel values (x > y)
	compMat = [[0 for i in range(8)] for j in range(8)]
	for i in range (0, x-1):
		for j in range (0, y):
			if imgMat[i][j] > imgMat[i][j+1]:
				compMat[i][j] = True
			else:
				compMat[i][j] = False

	#Convert to hash ---Hashing function by David Kofoed---
	hash_str = ""
	for difference in compMat:
		decimal_value = 0
		hex_str = []
		for i, value in enumerate(difference):
			if value:
				decimal_value += 2**(i % 8)
			if (i % 8) == 7:
				hex_str.append(hex(decimal_value)[2:].rjust(2, '0'))
				decimal_value = 0
		hash_str = hash_str +''.join(hex_str) + ' ' 
	return hash_str[:-1]

# Access all files in directory
#allfiles=os.listdir(os.getcwd())
#imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]


# Assuming all images are the same size, get dimensions of first image
#w,h=Image.open(imlist[0]).size
#N=len(imlist)

video1 = cv2.VideoCapture('videos/0A6HPR9NHFMF.mp4')
video2 = cv2.VideoCapture('videos/0OBAQSL3HY6B.mp4')
#0AJSX00QVPBI

w = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
N = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

w2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
h2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
N2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

#arr=np.zeros((h,w,3),np.float)
arr1=np.zeros((h,w),np.float)
arr2=np.zeros((h2,w2),np.float)
N = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(32,32))
#cl1 = clahe.apply(img)

while(True):#video1.isOpened()):# or video2.isOpened()):

	if video1.isOpened():
		ret1,frame1 = video1.read()
	if video2.isOpened():
		ret2,frame2 = video2.read()
	
	if ret1:
		frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		frame1 = clahe.apply(frame1)
		cv2.imshow('frame',frame1)
		imarr1=np.array(frame1,dtype=np.float)
		arr1=arr1+imarr1/N
	
	if ret2:
		frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		frame2 = clahe.apply(frame2)
		cv2.imshow('frame2',frame2)
		imarr2=np.array(frame2,dtype=np.float)
		arr2=arr2+imarr2/N2
	
	

	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break

	if not(video1.isOpened() and video2.isOpened()):
		cv2.destroyAllWindows()
		break


arr1=np.array(np.round(arr1),dtype=np.uint8)
arr2=np.array(np.round(arr2),dtype=np.uint8)

while True:

	cv2.imshow('average1',arr1)
	cv2.imshow('average2',arr2)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

out1=Image.fromarray(arr1,mode="L")
out2=Image.fromarray(arr2,mode="L")

print(getImageHash(out1))
print(getImageHash(out2))