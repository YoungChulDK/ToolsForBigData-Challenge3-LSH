import pylab
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2
import time
from skimage import color
from skimage import exposure
import imagehash as ih
from annoy import AnnoyIndex
from PIL import Image
import os
from sklearn.metrics import adjusted_rand_score

truth = [set(['NY0XRPCQX2J6', '5B15T46T75XM', 'QKPLUGBHWX1S', '90BP7NQLOZI8', 'H3ETKWH70OZ0', 'BWWQDUXMWDTU', '0J5OWQRLV2ZF', 'D0K9L1DTG1EQ', 'SRXWGC3XXJJO', '148X2AS0P7MP'])]

def rand_index(clusters):
	elems = list(set.union(*truth))

	# Index of Containing Set
	memory_truth = {}
	memory_clusters = {}
	def ics(element, set_list, set_list_name):
		if set_list_name == "truth":
			if element in memory_truth:
				return memory_truth[element]
		if set_list_name == "clusters":
			if element in memory_clusters:
				return memory_clusters[element]

		for c, s in enumerate(set_list):
			if element in s:
				if set_list_name == "truth":
					memory_truth[element] = c
				if set_list_name == "clusters":
					memory_clusters[element] = c
				return c

	x = map(lambda e: ics(e, clusters, 'clusters'), elems)
	y = map(lambda e: ics(e, truth, 'truth'), elems)

	return adjusted_rand_score(x,y)


def compute_avg(filename):
	vid = imageio.get_reader(filename,  'ffmpeg')
	skip = 4	
	N = int(vid.get_length()/skip)
	w,h = vid.get_meta_data()['size']

	avg = np.zeros((h,w),np.float)
	
	clahe = cv2.createCLAHE()#clipLimit=40.0, tileGridSize=(32,32))	

	for i,im in enumerate(vid):

		if i%skip==0:
			
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = clahe.apply(im)
			im = np.array(im,dtype=np.float)

			avg = avg+im/N

	avg=np.array(np.round(avg),dtype=np.uint8)
	return avg

def create_vector(s):
	vec = []
	for c in s:
		vec.append(float(ord(c)))
	return vec

di = '/Users/Henrik/Desktop/big_data/challenge3/small_set2'
names = []
hashes = []

t = AnnoyIndex(16)
files = os.listdir(di)
idx = 0
for f in files:
	if f.endswith('.mp4'):
		
		print(idx)
		#print(os.path.splitext(f)[0])
		names.append(os.path.splitext(f)[0])
		avg = compute_avg('small_set/'+f)
		h = ih.phash(Image.fromarray(avg))
		v = create_vector(str(h))
		#print(str(v))
		t.add_item(idx, v)
		idx+=1
		hashes.append(h)
	

print('building trees')
t.build(10)
t.save('test1.ann')
print('done building trees')
print(names[0])
nns = t.get_nns_by_item(0, 10) # will find the 10 nearest neighbors
clusters = set()
cl = []
for i,n in enumerate(nns):
	clusters.add(names[i])
	print(names[i])
cl.append(clusters)
print(rand_index(cl))

'''
filename1 = 'videos/DMDR1U2RA7VN.mp4'
filename2 = 'videos/K29U1709EA5R.mp4'
avg1 = compute_avg(filename1)
avg2 = compute_avg(filename2)
h1 = ih.phash(Image.fromarray(avg1))
h2 = ih.phash(Image.fromarray(avg2))
print(h1)
print(h2)
print(h1-h2)
'''








