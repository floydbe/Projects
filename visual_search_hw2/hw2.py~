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

def RGB_to_gray(im):
	a = im.shape[0]
	b = im.shape[1]
	result = np.zeros((a,b), dtype='uint8')
	for x in range(a):
		for y in range(b):
			result[x][y] = 0.299*im[x][y][0] + \
			0.587*im[x][y][1] + 0.114*im[x][y][2]
	return result

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

'''
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
'''
# Question 2
for i in range(1,24):
	a = imread("Test" + str(i) + ".png")
	b = RGB_to_gray(a)
	x = detect_arrow(a)
	y = detect_circle(b)
	print "image", i, x, y

