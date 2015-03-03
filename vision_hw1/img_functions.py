from PIL import Image, ImageDraw
import numpy as np
import pylab as plt
from scipy import ndimage
from scipy import signal
import matplotlib.cm as cm
import math
import time
from operator import itemgetter
from skimage.draw import circle_perimeter
import cv2
	
def resize_half(im):
	new = np.zeros((im.shape[0]/2, im.shape[1]/2))
	for a in range(new.shape[0]):
		for b in range(new.shape[1]):
			new[a][b] = im[2*a][2*b]
	return new
	
def resize(im,sz):	
	pil_im = Image.fromarray(np.uint8(im))
	return np.array(pil_im.resize(sz), dtype = 'int64')

def threshold(im, t):
	new_im = np.copy(im)
	#new_im[new_im > t] = 255
	new_im[new_im <= t] = 0
	return new_im

def gaussian(im, sigma):
	return ndimage.filters.gaussian_filter(im,sigma)

def gradient(im):
	xGradient = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	yGradient = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
	dx = ndimage.convolve(im, xGradient)
	dy = ndimage.convolve(im, yGradient)
	tot_grad = np.sqrt(np.square(np.absolute(dx)) + np.square(np.absolute(dy)))
	theta = np.arctan2(dy, dx)
	return (tot_grad, theta, np.absolute(dx), np.absolute(dy))
	
def display_hist(im):
	plt.hist(im.flatten(),bins=64)
	plt.show()

def hist_eq(im, nbr_bins=256):
	imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
	cdf = imhist.cumsum()
	cdf = cdf*255/cdf[-1]
	im2 = np.interp(im.flatten(),bins[:-1],cdf)
	return im2.reshape(im.shape), cdf
	
def non_max_suppression(grad, theta):
	for i in np.nditer(theta, op_flags=['readwrite']):
		if i < math.pi/8:
			i[...] = 0
		elif i < 3*math.pi/8:
			i[...] = 1
		elif i < 5*math.pi/8:
			i[...] = 2
		elif i < 7*math.pi/8:
			i[...] = 3
		else:
			i[...] = 0
	a,b = theta.shape
	for x in range(a):
		for y in range(b):
			if theta[x][y] == 2:
				try:
					if grad[x][y] < grad[x+1][y]:
						grad[x][y] = 0
					elif grad[x][y] < grad[x-1][y]:
						grad[x][y] = 0
				except:
					grad[x][y] = 0	
			elif theta[x][y] == 1:
				try:
					if grad[x][y] < grad[x+1][y+1]:
						grad[x][y] = 0
					elif grad[x][y] < grad[x-1][y-1]:
						grad[x][y] = 0
				except:
					grad[x][y] = 0 
			elif theta[x][y] == 2:
				try:
					if grad[x][y] < grad[x][y+1]:
						grad[x][y] = 0
					elif grad[x][y] < grad[x][y-1]:
						grad[x][y] = 0
				except:
					grad[x][y] = 0
			elif theta[x][y] == 3:
				try:
					if grad[x][y] < grad[x-1][y+1]:
						grad[x][y] = 0
					elif grad[x][y] < grad[x+1][y-1]:
						grad[x][y] = 0
				except:
					grad[x][y] = 0
	return grad
	
def hysteresis(grad, low_t, high_t):
	a,b = grad.shape
	visited = np.zeros((a,b), dtype='bool')
	above_high_t = np.zeros((a,b), dtype='bool')
	for x in range(a):
		for y in range(b):
			if grad[x][y] > high_t:
				above_high_t[x][y] = True
	final = np.copy(above_high_t)
	for m in range(a):
		for n in range(b):
			if above_high_t[m][n] == True:
				if visited[m][n] == False:
					visited[m][n] = True
					final, visited = explore(final, grad, low_t, high_t, m, n, visited)
	return final
	
def explore(final, grad, low_t, high_t, i, j, visited):
	try:
		if (visited[i-1][j] == False) and (grad[i-1][j] > low_t):
			final[i-1][j] = True
			final, visited = explore(final, grad, low_t, high_t, i-1, j, visited)
		visited[i-1][j] = True
	except: pass
	try:
		if (visited[i+1][j] == False) and (grad[i+1][j] > low_t):
			final[i+1][j] = True
			visited[i+1][j] = True
			final, visited = explore(final, grad, low_t, high_t, i+1, j, visited)
		visited[i+1][j] = True
	except: pass
	try:
		if (visited[i][j-1] == False) and (grad[i][j-1] > low_t):
			final[i][j-1] = True
			visited[i][j-1] = True
			final, visited = explore(final, grad, low_t, high_t, i, j-1, visited)
		visited[i][j-1] = True
	except: pass
	try:
		if (visited[i][j+1] == False) and (grad[i][j+1] > low_t):
			final[i][j+1] = True
			visited[i][j+1] = True
			final, visited = explore(final, grad, low_t, high_t, i, j+1, visited)
		visited[i][j+1] = True
	except: pass
	try:
		if (visited[i-1][j-1] == False) and (grad[i-1][j-1] > low_t):
			final[i-1][j-1] = True
			visited[i-1][j-1] = True
			final, visited = explore(final, grad, low_t, high_t, i-1, j-1, visited)
		visited[i-1][j-1] = True
	except: pass
	try:
		if (visited[i-1][j+1] == False) and (grad[i-1][j+1] > low_t):
			final[i-1][j+1] = True
			visited[i-1][j+1] = True
			final, visited = explore(final, grad, low_t, high_t, i-1, j+1, visited)
		visited[i-1][j+1] = True
	except: pass
	try:
		if (visited[i+1][j+1] == False) and (grad[i+1][j+1] > low_t):
			final[i+1][j+1] = True
			visited[i+1][j+1] = True
			final, visited = explore(final, grad, low_t, high_t, i+1, j+1, visited)
		visited[i+1][j+1] = True
	except: pass
	try:
		if (visited[i+1][j-1] == False) and (grad[i+1][j-1] > low_t):
			final[i+1][j-1] = True
			visited[i+1][j-1] = True
			final, visited = explore(final, grad, low_t, high_t, i+1, j-1, visited)
		visited[i+1][j-1] = True
	except: pass
	return final, visited

def canny_edge(im, sigma):
	blurred = gaussian(im, sigma)
	save(blurred, "checker_blur.jpg")
	(grad, theta, dx, dy) = gradient(blurred)
	save(dx,"checker_dx.jpg")
	save(dy,"checker_dy.jpg")
	save(grad,"checker_grad.jpg")
	suppressed = non_max_suppression(grad, theta)
	save(suppressed, "checker_suppressed.jpg")
	low_t = 0.66*np.median(im)
	high_t = 1.33*np.median(im)
	final = hysteresis(suppressed, low_t, high_t)
	save(final, "checker_edge.jpg")
	return final

def display(im):
	plt.imshow(im, cmap = cm.Greys_r)
	plt.show()

def display2(im1, im2):
	f = plt.figure()	
	f.add_subplot(1,2,1)
	plt.imshow(im1, cmap = cm.Greys_r)
	f.add_subplot(1,2,2)
	plt.imshow(im2, cmap = cm.Greys_r)
	plt.show()

def save(im, filename):
	rescaled = (255.0 / im.max() * (im - im.min())).astype(np.uint8)
	#rescaled = np.uint8(im)
	im = Image.fromarray(rescaled)
	im.save(filename)

def filter_by_eig(l, result, thresh):
	new_l = []
	for el in l:
		if el[0] > thresh:
			new_l.append(el)
			result[el[1]][el[2]] = el[0]
	return new_l, result		

def harris_corner(im, sigma, n_size = 16):
	candidates = []
	blurred = gaussian(im, sigma)
	grad, theta, dx, dy = gradient(blurred)
	a,b = im.shape
	result = np.zeros((a,b))
	for x in range(a):
		for y in range(b):
			covar = np.zeros((2,2))
			for i in range(-1*n_size, n_size+1):
				for j in range(-1*n_size, n_size+1):
					try:
						covar[0][0] += dx[x+i][y+j] * dx[x+i][y+j] 
						covar[0][1] += dx[x+i][y+j] * dy[x+i][y+j]
						covar[1][0] += dx[x+i][y+j] * dy[x+i][y+j]
						covar[1][1] += dy[x+i][y+j] * dy[x+i][y+j]
					except:
						pass
			eig_val, eig_vect = np.linalg.eig(covar)
			eig = eig_val[0]
			if eig_val[1] < eig: 
				eig = eig_val[1]
			candidates.append([eig,x,y])
	eigs = np.array([i[0] for i in candidates])
	threshold = (2 * np.median(eigs) + np.amax(eigs)) / 3
	candidates, result = filter_by_eig(candidates, result, threshold)
	candidates.sort(key = itemgetter(0), reverse = True) 
	for e,y,x in candidates:
		for i in range(-1*n_size, n_size+1):
			for j in range(-1*n_size, n_size+1):
				try:
					if result[x+i][y+j] < e:
						result[x+i][y+j] = 0
						result[x][y] = 255
				except:
					print "OOB"
	return result

def enlarge_points(im, area = 3):
	a,b = im.shape
	result = np.zeros((a,b))
	for x in range(a):
		for y in range(b):
			if not (im[x][y] == 0):
				for p in range(-(area-1)/2, (area-1)/2 + 1):
					for q in range(-(area-1)/2, (area-1)/2 + 1):
						try:
							result[x+p][y+q] = im[x][y]
						except:
							pass
	return result

def is_extrema(x,y,lower,middle,upper):
	minimum = True
	maximum = True
	
	height,width = middle.shape
	comparison = middle[x][y]
	for a in [-1,0,1]:
		for b in [-1,0,1]:
			if (not maximum) and (not minimum):
				return False
			try:
				if upper[x+a][y+a] >= comparison:
					maximum = False
				elif lower[x+a][y+b] >= comparison:
					maximum = False
				elif (not ((a == 0) and (b == 0))) and (middle[x+a][y+b] >= comparison):
					maximum = False
			
				if upper[x+a][y+a] <= comparison:
					minimum = False
				elif lower[x+a][y+b] <= comparison:
					minimum = False
				elif (not ((a == 0) and (b == 0))) and (middle[x+a][y+b] <= comparison):
					minimum = False
			except:
				maximum = minimum = False
	return (maximum or minimum)
		
def calc_extrema(dogs):
	width, height = dogs[0].shape
	extrema = []
	for x in range(width):
		for y in range(height):
			for a in range(2, len(dogs)):
				is_ex = is_extrema(x,y, dogs[a], dogs[a-1], dogs[a-2])
				if is_ex:
					extrema.append([x,y,a-1])
	return extrema

def draw_circle(im, x, y, r):
	rr, cc = circle_perimeter(int(x),int(y),int(r*3))
	try:
		im[rr, cc] = 255
	except:
		pass
	return im
				
def sift(img, sigma = 1.6, num_scales = 5, num_octaves = 4, k = math.sqrt(2)):
	im = img.copy()
	keypoints = []
	wid, hei = im.shape
	result = np.zeros((wid,hei))
	for o in range(num_octaves):
		octave = []
		for s in range(num_scales):
			octave.append(gaussian(im, (k**s)*sigma))
		dogs = []
		for x in range(len(octave)-1):
			dogs.append(octave[x+1] - octave[x])
		extrema = calc_extrema(dogs)
		for a,b,c in extrema:
			c = math.pow(k,o)*math.pow(k,c)*sigma
			try:
				a = a * (2**o)
				b = b * (2**o)
				result[a][b] = c
				keypoints.append([a,b,c])
			except:
				pass
		im = octave[len(octave)-1-2]
		im = resize_half(im)
	harris = enlarge_points(harris_corner(img, sigma),9)
	for p,q,r in keypoints:
		if harris[p][q] == 255:
			img = draw_circle(img, p, q, r)
	return img

def draw_corners(corn, img):
	a,b = corn.shape
	save(img,"checker_corn1.jpg")
	for x in range(a):
		for y in range(b):
			if corn[x][y] == 255:
				l_x = x-2
				r_x = x+2
				t_y = y-2
				b_y = y+2
				cv2.rectangle(img,(l_x,t_y),(r_x,b_y),(256,0,0))
	save(img, "checker_corn2.jpg")

keystring = 'checker'
a = Image.open('Images/' + keystring + '.jpg').convert('L')
b = np.array(a, dtype = 'int64')
c = np.array(Image.open('Images/' + keystring + '.jpg'))
#edge = canny_edge(b, 1.25)
#save(edge, keystring + '_edge.jpg')
#corn = harris_corner(b, 2, 3)
#draw_corners(corn, c)
#corn = enlarge_points(corn,9)
#save(corn, keystring + '_corn.jpg')
s = sift(b)
save(s, keystring + '_sift.jpg')
