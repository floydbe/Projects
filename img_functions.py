from PIL import Image
import numpy as np
import pylab as plt
from scipy import ndimage
from scipy import signal
import matplotlib.cm as cm
import math
import time
from operator import itemgetter
	
def imresize(im,sz):
	pil_im = Image.fromarray(uint8(im))
	return np.array(pil_im.resize(sz))

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
	print "beginning edge detection"
	print "blurring image with sigma =", sigma
	blurred = gaussian(im, sigma)
	print "calculating gradient"
	(grad, theta, dx, dy) = gradient(blurred)
	print "applying non-maximum suppression"
	suppressed = non_max_suppression(grad, theta)
	low_t = 0.66*np.median(im)
	high_t = 1.33*np.median(im)
	print "applying hysteresis with (low_t, high_t) =", (low_t,high_t)
	final = hysteresis(suppressed, low_t, high_t)
	print "edge detection complete"
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
	im = Image.fromarray(rescaled)
	im.save(filename)

def filter_by_eig(l, result, thresh):
	new_l = []
	for el in l:
		if el[0] > thresh:
			new_l.append(el)
			result[el[1]][el[2]] = el[0]
	return new_l, result		

def harris_corner(im, sigma, n_size = 4):
	print '\nstarting corner detection'
	candidates = []
	blurred = gaussian(im, sigma)
	grad, theta, dx, dy = gradient(blurred)
	a,b = im.shape
	result = np.zeros((a,b))
	print 'calculating covarience matrices'
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
	print 'applying threshold:', threshold
	candidates, result = filter_by_eig(candidates, result, threshold)
	candidates.sort(key = itemgetter(0), reverse = True) 
	print 'expanding maximal points'
	for e,x,y in candidates:
		for i in range(-1*n_size, n_size+1):
			for j in range(-1*n_size, n_size+1):
				try:
					if result[x+i][y+j] < e:
						if (abs(i) <= 1) and (abs(i) <= 1):
							result[x+i][y+j] = e
						else:
							result[x+i][y+j] = 0.0
				except:
					pass
	print 'corner dectection complete'
	return result

def sift(im, sigma = 1.6, num_scales = 5, num_octaves = 4, k = math.sqrt(2)):
	im = gaussian(im, sigma)
	

keystring = 'mandrill'
a = Image.open('Images/' + keystring + '.jpg').convert('L')
b = np.array(a, dtype = 'int64')
edge = canny_edge(b, 1.25)
save(edge, keystring + '_edge.jpg')
corn = harris_corner(b, 2, 3)
save(corn, keystring + '_corn.jpg')
