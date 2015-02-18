import numpy as np
from scipy import misc, ndimage, stats
import pylab as plt
import math
from skimage import transform, feature, filter, color, draw
import cv2
import cv2.cv as cv

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

def my_convolve(h,i):
	a,b = i.shape
	result = np.zeros((a,b))
	for x in range(a):
		for y in range(b):
			for p in [-1,0,1]:
				for q in [-1,0,1]:
					if x+p >= 0 and x+p < a and y+q >= 0 and y+q < b:
						result[x][y] += h[p+1][q+1] * i[x+p][y+q]
					
	return result	

# logic for determining if a list of lines forms an arrow
def is_arrow(angles):
	angles.sort()
	for x in range(len(angles)-1):
		if abs(angles[x+1] - angles[x]) < 0.05:
			for a in angles:
				if abs(abs(a - angles[x])-math.pi/4) < 0.1:
					for b in angles:
						if abs(abs(a-b)-math.pi/2) < 0.1:
							return True 
	
	return False
	
# detects arrows in an image by using Hough Transform to detect lines,
# then pass off to another function to determine if they are arranged 
# as an arrow
def detect_arrow(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	angles = []
	try:
		lines = cv2.HoughLines(edges,1,np.pi/90,58)
		for rho,theta in lines[0]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			angles.append(theta)
	
			cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
		
		return is_arrow(angles)
		
	except:
		return False
# detect circles in an image. Returns true if a circle satisfying 
# certain parameters/thresholds is found. Returns false otherwise.
def detect_circle(img):
	cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
	try:
		circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,2,50,
				                    param1=50,param2=225,minRadius=5,maxRadius=300)
		x = circles[0] # forces an exception if circles is empty (there were no circles)
		return True
	except:
		return False

# Question 1
a = imread("cameraman.tif")
a = np.int64(a)
h1 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
h2 = np.array([[-1,-1,-1],[-1, 8, -1],[-1,-1,-1]])
y1 = my_convolve(h1,a)
y2 = my_convolve(h2,a)
imsave(y1, "conv_h1.tif")
imsave(y2, "conv_h2.tif")
d1 = y1-a
d2 = y2-a
imsave(d1, "d1.tif")
imsave(d2, "d2.tif")
s1 = y1+a
s2 = y2+a
imsave(s1, "s1.tif")
imsave(s2, "s2.tif")

# Question 2
for i in range(1,24):
	a = imread("visual_search_hw2/Test" + str(i) + ".png")
	b = RGB_to_gray(a)
	x = detect_arrow(a)
	y = detect_circle(b)
	print "image", i, x, y

