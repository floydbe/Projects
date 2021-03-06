import os
import cv2
import numpy as np
import Image
from scipy import misc, ndimage
import pylab as plt
import random
import math

testA_im_loc = "test/"
testA_data_loc = "testA_data.txt"

def imread(filename):
	return np.array(Image.open(filename).convert('L')),np.array(Image.open(filename))
	
def imsave(im, filename):
	misc.imsave(filename, im)

def get_imlist(path, filetype):
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(filetype)]

def imresize(im, size):
	im = Image.fromarray(np.uint8(im))
	return np.array(im.resize(size))

def gaussian(im, sigma):
	return ndimage.filters.gaussian_filter(im,sigma)

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
			i,ci = imread(im)
			cropped = i[miny-y_scale:maxy+y_scale,minx-x_scale:maxx+x_scale]
			cropped = imresize(cropped, (12,12))
			faces.append(cropped)
face_mean = combine(faces)
imsave(face_mean,"face.jpg")

# Build up list of not-face patches
not_faces = []
while len(not_faces) < 100:
	for im in imlist:
		i,ci = imread(im)
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

def create_gaussian(t_set,mean):
	cov = np.zeros((100,144))

	for a in range(100):
		for b in range(144):
			cov[a][b] = t_set[a].flatten()[b]-mean.flatten()[b]
	cov = np.matrix(cov)
	cov_t = np.matrix.transpose(cov)
	A = ( cov_t * cov ) / 144.0
	U, s, V = np.linalg.svd(A)
	tau = 10.0
	k = 0
	det = 1
	for sv in s:
		if sv > tau:
			det *= sv
			k += 1
	u = np.matrix(U[:,:k])
	s = s[:k]
	u_t = np.matrix.transpose(u)
	s = np.matrix(np.diag(s))
	e = u*s*u_t
	s_inv = np.linalg.inv(s)
	e_inv = u*s_inv*u_t
	return e_inv,k,det
	
def score(patch,mean,e_inv,k,det):
	x = np.matrix(patch.reshape((144,1))) - np.matrix(mean.reshape((144,1)))
	return np.exp((-1.0/2.0)*np.matrix.transpose(x) * e_inv * x) * 1.0e100 / (math.sqrt(((2*math.pi)**k)*det))


def gaussian_face(img,pi):
	a,b = img.shape
	result = np.zeros((a,b))
	face_gaussian = create_gaussian(faces, face_mean)
	not_face_gaussian = create_gaussian(not_faces, not_face_mean)

	for y in range(a-12):
		for x in range(b-12):
			patch = img[x:x+12,y:y+12]
			f = pi * score(patch, face_mean, face_gaussian[0], face_gaussian[1], face_gaussian[2])
			nf = (1-pi) * score(patch, not_face_mean, not_face_gaussian[0], not_face_gaussian[1], not_face_gaussian[2])
			if f[0][0] > nf[0][0]:
				result[x][y] = f
	return result

def scaled_gaussian(im, mu):
	a,b = im.shape
	result = np.zeros((a,b))
	scale = np.zeros((a,b))
	step = 0.1
	mult = 0
	p, q = int(a-(a*mult*step)), int(b-(b*mult*step))
	new = np.copy(im)
	while (a > 24 and b > 24 and mult < 8):
		print "examining at", (1-mult*step)*100, "%"
		print "size:", p, q
		intermediate = gaussian_face(new, mu)
		for y in range(p):
			for x in range(q):
				if intermediate[x][y] > result[int(x+(x*mult*step)+0.5)][int(y+(y*mult*step)+0.5)]:
					print (x,y), "->" , (int(x+(x*mult*step)+0.5) , int(y+(y*mult*step)+0.5))
					result[int(x+(x*mult*step)+0.5)][int(y+(y*mult*step)+0.5)] = intermediate[x][y]
					scale[int(x+(x*mult*step)+0.5)][int(y+(y*mult*step)+0.5)] = mult
		mult += 1
		p, q = int(a-(a*mult*step)), int(b-(b*mult*step))
		new = gaussian(new, 1.5)
		new = imresize(new, (p,q))
		
	return result, scale

''' The evaluation function used in the linear classifier. Note that I had to
	divide the exponent by 10000000 to counter an overflow issue. This does not
	affect its performance because it uses the same scalar every time.
'''
def g(x,w):
	x = x.flatten()
	w = w.flatten()
	exponent = -1.0 * np.dot(x,w)
	score = 1.0/(1.0 + np.exp(exponent/10000000.0)) 
	return score

""" The function to train the linear classifier. Takes observations (patches),
	classifications (face or not face), mu (learning rate), and optionallly 
	convergence criteria. Returns w, the vector used to to calculate the score
	of future candidate patches.
"""
def train_linear(observations, classifications, mu, convergence=1.0e-2, max_iterations=300):
	a,b = observations[0].shape
	w = np.zeros((a*b+1,1))
	for i in range(len(observations)):
		observations[i] = np.reshape(observations[i],(144,1))
		observations[i] = np.vstack((observations[i], [1]))

	iterations = 0
	while True:
		summation = np.zeros(w.shape)
		for o,c in zip(observations, classifications):
			summation += (c-g(o,w))*o

		scale_sum = mu*summation
		w_delta_sum = np.add.reduce(scale_sum)
		w = w + scale_sum

		if math.fabs(w_delta_sum) <= convergence:
			break
		iterations += 1
		if iterations > max_iterations:
			break
	return w


""" Takes an image, a list of observations (patches), a list of classifications
	(face or not face), and mu (learning rate). Returns a numpy array in shape
	of image with pixels. If pixel value is not 0, location is a possible face
"""
def linear_face(img, w):
	a,b = img.shape
	result = np.zeros((a,b))
	for y in range(a-12):
		for x in range(b-12):
			patch = img[x:x+12,y:y+12]
			patch = np.reshape(patch, (144,1))
			patch = np.vstack((patch, [1]))
			score = g(patch,w)
			if score > 0.5:
				result[x][y] = score
	return result
				
def scaled_linear(im,w ):
	a,b = im.shape
	result = np.zeros((a,b))
	scale = np.zeros((a,b))
	step = 0.1
	mult = 0
	p, q = int(a-(a*mult*step)), int(b-(b*mult*step))
	new = np.copy(im)
	while (a > 24 and b > 24 and mult < 8):
		print "examining at", (1-mult*step)*100, "%"
		print "size:", p, q
		intermediate = linear_face(new, w)
		for y in range(p):
			for x in range(q):
				if intermediate[x][y] > result[int(x+(x*mult*step)+0.5)][int(y+(y*mult*step)+0.5)]:
					print (x,y), "->" , (int(x+(x*mult*step)+0.5) , int(y+(y*mult*step)+0.5)), intermediate[x][y], mult
					result[int(x+(x*mult*step)+0.5)][int(y+(y*mult*step)+0.5)] = intermediate[x][y]
					scale[int(x+(x*mult*step)+0.5)][int(y+(y*mult*step)+0.5)] = mult
		mult += 1
		p, q = int(a-(a*mult*step)), int(b-(b*mult*step))
		imsave(new, "mult" + str(mult) + ".jpg")
		new = gaussian(new, 1)
		new = imresize(new, (p,q))
		
	return result, scale

def non_max_suppression(result,m=3):
	a,b = result.shape
	for y in range(m,a-m):
		for x in range(m,b-m):
			for yy in range(-m,m+1):
				for xx in range(-m,m+1):
					if result[x+xx][y+yy] > result[x][y]:
						result[x][y] = 0 
	return result

def draw_boxes(suppressed, scale, cimg):
	a,b = suppressed.shape
	for y in range(a):
		for x in range(b):
			if suppressed[x][y] != 0:
				cv2.rectangle(cimg,(y,x),(y+12+(int(12*0.1*scale[x][y]+0.5)),x+12+(int(12*0.1*scale[x][y]+0.5))),(0,0,255))
				print "drawing box at", y, x, "at scale", scale[x][y], "of size", 12 + int(12*0.1*scale[x][y]+0.5)
	return cimg

im,cim = imread("test_input/solidbg-different-sizes.gif")
'''
g_result,g_scale = scaled_gaussian(im,0.1)
suppressed = g_result#non_max_suppression(g_result,12)
result_img = draw_boxes(suppressed,g_scale,cim)
imsave(result_img, "face_detect.jpg")
'''
observations = faces[:100] + not_faces[:100]
classifications = [1 for i in range(100)] + [0 for j in range(100)]
w = train_linear(observations,classifications, 0.5)
l_result, scale = scaled_linear(im, w)
print "performing non-maximal suppression"
suppressed = non_max_suppression(l_result,18)
print "drawing boxes"
l_result_img = draw_boxes(suppressed, scale, cim)
imsave(l_result_img, "face_detect_linear.jpg")

