import os
import cv2
import numpy as np
import Image
from scipy import misc
import pylab as plt
import random
import math

testA_im_loc = "test/"
testA_data_loc = "testA_data.txt"

def imread(filename):
	return np.array(Image.open(filename).convert('L'))
	
def imsave(im, filename):
	misc.imsave(filename, im)

def get_imlist(path, filetype):
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(filetype)]

def imresize(im, size):
	im = Image.fromarray(np.uint8(im))
	return np.array(im.resize(size))

''' Takes a list of intensity images (numpy arrays) which must 
	all have the same dimensions and computes a new image that 
	is the average of each of the input images (pixel by pixel). 
'''
def combine(im_list):
	a,b = im_list[0].shape			# get the dimensions of the images
	result = np.zeros((a,b))		# create result images of the same size
	for i in im_list:				# loop through list of images,
		for x in range(a):			# adding pixel value to result image
			for y in range(b):
				result[x][y] += i[x][y]
	return result / len(im_list)	# divide the sum array by the number of images


def create_collage(faces, not_faces):
	height = 10
	width = 10
	l_result = np.zeros((240,1))
	for i in range(width):
		column = imresize(faces[height*i], (24,24))	
		for j in range(height-1):
			column = np.concatenate((column,imresize(faces[height*i+j+1], (24,24))), axis=0)
		l_result = np.concatenate((l_result,column), axis=1)
	
	r_result = np.zeros((240,1))
	for i in range(width):
		column = imresize(not_faces[height*i], (24,24))	
		for j in range(height-1):
			column = np.concatenate((column,imresize(not_faces[height*i+j+1], (24,24))), axis=0)
		r_result = np.concatenate((r_result,column), axis=1)
	result = np.concatenate((l_result[:,1:], r_result[:,1:]), axis=1)
	
	imsave(result, "collage.jpg")


f = open(testA_data_loc, 'r')
face_locs = f.readlines()
f.close()

parsed = []
for face in face_locs:
	parsed.append(face.split())
for p in parsed:
	for idx in range(1,13):
		p[idx] = int(float(p[idx]))

imlist = get_imlist(testA_im_loc, '.gif')

# Build up list of face patches
faces = []
for im in imlist:
	for line in parsed:
		if testA_im_loc + line[0] == im:
			miny = min(line[2],line[4],line[6],line[8],line[10],line[12])
			maxy = max(line[2],line[4],line[6],line[8],line[10],line[12])
			minx = min(line[1],line[3],line[5],line[7],line[9],line[11])
			maxx = max(line[1],line[3],line[5],line[7],line[9],line[11])
			y_ran = maxy - miny
			x_ran = maxx - minx
			ex_perc = 0.2
			y_scale = int(ex_perc * y_ran)
			x_scale = int(ex_perc * x_ran)
			i = imread(im)
			cropped = i[miny-y_scale:maxy+y_scale,minx-x_scale:maxx+x_scale]
			cropped = imresize(cropped, (12,12))
			faces.append(cropped)
face_mean = combine(faces)
imsave(face_mean,"face.jpg")

# Build up list of not-face patches
not_faces = []
while len(not_faces) < 100:
	for im in imlist:
		i = imread(im)
		r1 = random.randint(0,i.shape[0]-1)
		r2 = random.randint(0,i.shape[1]-1)
		try:
			crop = i[r1:r1+12,r2:r2+12]
			crop = imresize(crop, (12,12))
			not_faces.append(crop)
		except IndexError:
			print "out of bounds"
		except:
			print "unknown"
not_face_mean = combine(not_faces)
imsave(not_face_mean,"notface.jpg")

create_collage(faces,not_faces)

face_cov = np.zeros((144,144))

for a in range(144):
	for b in range(144):
		if a < 100 and b < 100:
			face_cov[a][b] = abs(faces[a].flatten()[b]-face_mean.flatten()[b])
U, s, V = np.linalg.svd(face_cov)
tau = 0.00001
k = 0
det = 1
for sv in s:
	if sv > tau:
		det *= sv
		k += 1
u = U[:,:k]
s = s[:k]
u_t = np.transpose(u)
s = np.diag(s)
e = np.dot(u, np.dot(s,u_t))
s_inv = np.linalg.inv(s)
e_inv = np.dot(u, np.dot(s_inv,u_t))

print e_inv,k

x = faces[0]
diff = np.abs(x.flatten()-face_mean.flatten())
print diff
face_score = np.exp((-1.0/2.0)*np.dot(diff,np.dot(e_inv,np.transpose(diff))))/(math.sqrt(((2*math.pi)**k)*det))

print face_score
