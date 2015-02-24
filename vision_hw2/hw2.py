import os
import cv2
import numpy as np
import Image
from scipy import misc
import pylab as plt
import random
import time

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

imsave(combine(faces),"face.jpg")

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

imsave(combine(not_faces),"notface.jpg")
