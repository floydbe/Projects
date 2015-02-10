import numpy as np
from scipy import misc, ndimage, stats
import pylab as plt
import math
from skimage import transform, feature, filter, color, draw

def imread(filename):
	return misc.imread(filename)

def imsave(im, filename):
	misc.imsave(filename, im)

def imshow(im_list):
	l = len(im_list)
	fig, ax = plt.subplots(ncols=l)
	for i in range(l):
		ax[i].imshow(im_list[i],cmap = 'gray')
		ax[i].axis('off')
	plt.show()

def display_hist(im):
	plt.hist(im.flatten(),bins=64)
	plt.show()

def hist_eq(im, nbr_bins=256):
	imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
	cdf = imhist.cumsum()
	cdf = cdf*255/cdf[-1]
	im2 = np.interp(im.flatten(),bins[:-1],cdf)
	return im2.reshape(im.shape), cdf

def RGB_to_gray(im):
	a = im.shape[0]
	b = im.shape[1]
	result = np.zeros((a,b), dtype='uint8')
	for x in range(a):
		for y in range(b):
			result[x][y] = 0.299*im[x][y][0] + \
			0.587*im[x][y][1] + 0.114*im[x][y][2]
	return result

def gray_to_RGB(im):
	a,b = im.shape
	result = np.zeros((a,b,3),dtype='uint8')
	for x in range(a):
		for y in range(b):
			val = im[x][y]
			result[x][y][0] = val
			result[x][y][1] = val
			result[x][y][2] = val
	return result
		
def resize(im, s=2):
	new = np.zeros((im.shape[0]/s, im.shape[1]/s))
	for a in range(new.shape[0]):
		for b in range(new.shape[1]):
			new[a][b] = im[s*a][s*b]
	return new

def separate_bg(im):
	bg = im[0][0]
	a,b = im.shape
	result = np.zeros((a,b))
	for x in range(a):
		for y in range(b):
			if im[x][y] == bg:
				result[x][y] = 0
			else:
				result[x][y] = 255
	return result

def threshold_RGB(imageAr):
	balanceAr = []
	newAr = np.copy(imageAr)
	
	for eachrow in imageAr:
		for eachpix in eachrow:
			avgNum = reduce(lambda x, y: x + y,eachpix[:3])/3
			balanceAr.append(avgNum)
	balance = reduce(lambda x, y: x + y,balanceAr)/len(balanceAr)
	
	for row in newAr:
		for pix in row:
			if reduce(lambda x, y : x + y,pix[:3])/3 > balance:
				pix[0] = 255
				pix[1] = 255
				pix[2] = 255
			else:
				pix[0] = 0
				pix[1] = 0
				pix[2] = 0
	return newAr

def threshold_binary(im, t):
	new_im = np.zeros((im.shape), dtype='uint8')
	new_im[new_im > t] = 255
	new_im[new_im <= t] = 0
	return new_im

def threshold_filter(im, t):
	new_im = np.copy(im)
	new_im[new_im <= t] = 0
	return new_im

def gaussian(im, sigma = 2):
	return ndimage.filters.gaussian_filter(im,sigma)

def gradient(im):
	xGradient = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	yGradient = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
	im = im.astype(np.int64)
	dx = ndimage.convolve(im, xGradient)
	dy = ndimage.convolve(im, yGradient)
	tot_grad = np.sqrt(np.square(np.absolute(dx)) + np.square(np.absolute(dy)))
	theta = np.arctan2(dy, dx)
	return (tot_grad, theta, np.absolute(dx), np.absolute(dy))

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
	for x in [-1,0,1]:
		for y in [-1,0,1]:
			try:
				if (visited[i+x][j+y] == False) and (grad[i+x][j+y] > low_t):
					final[i+x][j+y] = True
					visited[i+x][j+y] = True
					final, visited = explore(final, grad, low_t, high_t, i+x, j+y, visited)
				else:
					visited[i+x][j+y] = True
			except:
				pass
	return final, visited

def canny_edge(im, sigma = 1.25):
	blurred = gaussian(im, sigma)
	(grad, theta, dx, dy) = gradient(blurred)
	suppressed = non_max_suppression(grad, theta)
	low_t = 0.66*np.median(im)
	high_t = 1.33*np.median(im)
	final = hysteresis(suppressed, low_t, high_t)
	final = (255.0 / final.max() * (final - final.min())).astype(np.uint8)
	return final

def hough_circles(im):
	image = np.copy(im)
	edges = filter.canny(image)
	# Detect two radii
	hough_radii = np.arange(im.shape[1]/16, im.shape[0]/2, 2)
	hough_res = transform.hough_circle(edges, hough_radii)

	centers = []
	accums = []
	radii = []

	for radius, h in zip(hough_radii, hough_res):
		# For each radius, extract two circles
		peaks = feature.peak_local_max(h, num_peaks=2)
		centers.extend(peaks)
		accums.extend(h[peaks[:, 0], peaks[:, 1]])
		radii.extend([radius, radius])
	
	image = color.gray2rgb(image)
	for idx in np.argsort(accums)[::-1][:1]:
		if accums[idx] > np.median(accums):
			center_x, center_y = centers[idx]
			radius = radii[idx]
			cx, cy = draw.circle_perimeter(center_y, center_x, radius)
			image[cy, cx] = (220, 20, 20)
	imshow([im,image])

def hough_lines(im):
	hspace, angles, dists = transform.hough_line(im)
	hspace, angles, dists = transform.hough_line_peaks(hspace, angles, dists)
	print hspace
    
for i in range(1,24):
	a = imread("visual_search_hw2/Test" + str(i) + ".png")
	b = RGB_to_gray(a)
	th = separate_bg(b)
	th_blur = gaussian(th,1)
	grad,theta,dx,dy = gradient(th_blur)
	inv = 255 - grad
	hough_circles(b)
	#imsave(grad, "Test" + str(i) + "_edge.png")

