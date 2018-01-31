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
import multiprocessing as mp
from sklearn.cluster import KMeans
from sklearn.cluster import k_means


truth = [set(['DMDR1U2RA7VN', 'K29U1709EA5R', 'D3NAY0YYFO4P', '58D4CGTDM5VX', 'ZLRB9DMOYSM9', 'J27VW94YYJRP', '77FOA4UNWD8Y', 'W0JQH817T6IE', 'OTXGMC3STDZ7', 'F4R4MW6W1BO8']), set(['NY0XRPCQX2J6', '5B15T46T75XM', 'QKPLUGBHWX1S', '90BP7NQLOZI8', 'H3ETKWH70OZ0', 'BWWQDUXMWDTU', '0J5OWQRLV2ZF', 'D0K9L1DTG1EQ', 'SRXWGC3XXJJO', '148X2AS0P7MP']), set(['YS0M2FXHFUKK', 'KASAZL3RPKK6', 'ZILSSCBC40IR', 'NEFEWA5CEPMW', '8DGQWN7D24RW', 'G1FQA6E96794', 'XNP69S9V9849', 'X5YBR7LX367U', '7INXG6910I57', 'W6G19WDE9FBN'])]

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

def get_file_chunks(files,n_cores):
	splits = len(files)/cores
	#chunks = zip(*[iter(files)]*4)
	chunks = [files[i:i+splits] for i in range(0,len(files),splits)]
	return chunks

def compute_avg(filename):
	vid = imageio.get_reader(filename,  'ffmpeg')
	
	skip = 4	
	N = int(vid.get_length()/skip)
	
	w,h = vid.get_meta_data()['size']

	avg = np.zeros((h,w),np.float)
	
	clahe = cv2.createCLAHE()#clipLimit=40.0, tileGridSize=(32,32))
		

	for i,im in enumerate(vid):

		if i%skip==0:
			'''
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im = clahe.apply(im)
			im = np.array(im,dtype=np.float)
			'''
			im = color.rgb2gray(im)
			im = exposure.equalize_hist(im)
			im = np.array(im*128,dtype=np.float)

			avg = avg+im/N

	avg=np.array(np.round(avg),dtype=np.uint8)
	return avg

def create_vector(s):
	vec = []
	for c in s:
		vec.append(float(ord(c)))
	return vec

def compute_hash(file_chunk,n):
	names = []
	hashes = []
	idx = 0

	for f in file_chunk:
		if f.endswith('.mp4'):
		
			#print(idx)
			print(os.path.splitext(f)[0])
			names.append(os.path.splitext(f)[0])
			
			avg = compute_avg('small_set/'+f)
			h = ih.phash(Image.fromarray(avg))
			v = create_vector(str(h))
			hashes.append(v)
			#print(str(v))
			#t.add_item(idx, v)
			idx+=1
	print('process number '+str(n)+' finished')
	return (names, hashes)
		
t1 = time.time()
direc = '/Users/Henrik/Desktop/big_data/challenge3/small_set'
files = os.listdir(direc)

cores = mp.cpu_count()
pool = mp.Pool(cores)
jobs = []
'''
file_chunks = get_file_chunks(files,cores)
print(len(file_chunks))

job_n = 1

for chunk in file_chunks:
	print('Process number '+str(job_n)+' created')
	jobs.append(pool.apply_async(compute_hash,(chunk,job_n)))
	job_n += 1

hash_arr = []
names = []
#wait for all jobs to finish
for job in jobs:
	result = job.get()
	
	for i in range(len(result[0])):
		hash_arr.append(result[1][i])
		names.append(result[0][i])
	

'''
hash_arr = compute_hash(files,1)
print(len(hash_arr[0]))
X = np.array(hash_arr[1])
names = hash_arr[0]


#X = np.array(hash_arr)
print('length of hash_arr '+str(len(hash_arr)))
print('shape of X '+str(X.shape))
Nclusters = 30

centroids, cls, inertia = k_means(X, Nclusters, verbose=False, max_iter=100, n_init=30)

clusters = [set() for i in range(Nclusters)]

print('clusters length')
print(len(clusters))
print('clusters type')
print(type(clusters[0]))

#names = hash_arr[0]
print('type of cls '+str(type(cls)))
for i in range(cls.size):
	clusters[cls[i]].add(names[i])

#for i,c in enumerate(cls):
#	print(c)
#	clusters[c].add(names[i])

for i,c in enumerate(clusters):
	print('cluster number '+str(i))
	print(c)
print(rand_index(clusters))
print('The process took '+str(time.time()-t1))

