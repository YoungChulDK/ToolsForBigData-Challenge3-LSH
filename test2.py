import pylab
import imageio
import numpy as np
import cv2
import time
from skimage import color
from skimage import exposure


t1 = time.time()
#filename = 'videos/1C4R01VC8HVS.mp4'
filename = 'videos/0ACLYF0HM2KX.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
skip = 4
N = int(vid.get_length()/skip)
w,h = vid.get_meta_data()['size']
print(w,h)

avg = np.zeros((h,w),np.float)
#print(type(vid))	

for i,im in enumerate(vid):
	#print('Mean of frame %i is %1.1f' % (i, im.mean()))
	if i%skip==0:
		im = color.rgb2gray(im)
		im = exposure.equalize_hist(im)
		im = np.array(im*128,dtype=np.float)
		avg = avg+im/N

avg=np.array(np.round(avg),dtype=np.uint8)
print(time.time()-t1)

lol = color.rgb2gray(vid.get_data(1))
lol = exposure.equalize_hist(lol)
print(np.amax(lol))
pylab.imshow(avg)
pylab.show()

while True:
	cv2.imshow('frame',lol)
	cv2.imshow('frame2',avg)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


'''
while True:

	#v2.imshow('average1',avg)
	pylab.imshow(avg)
	pylab.show()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
'''