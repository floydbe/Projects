import numpy as np
from scipy import misc, ndimage, stats
import pylab as plt
import math
import cv2
import cv2.cv as cv
import time

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
	
def detect_arrow(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	angles = []
	lmap = np.zeros(img.shape)
	try:
		lines = cv2.HoughLines(edges,1,np.pi/90,58)
		for rho,theta in lines[0]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 2000*(-b))
			y1 = int(y0 + 2000*(a))
			x2 = int(x0 - 2000*(-b))
			y2 = int(y0 - 2000*(a))
			angles.append([theta,x1,y1,x2,y2])
		
		for a in angles:
			angles1 = angles[:]
			angles1.remove(a)
			for b in angles1:
				angles2 = angles1[:]
				angles2.remove(b)
				if abs(a[0] - b[0]) < 0.05:
					for c in angles2:
						angles3 = angles2[:]
						angles3.remove(c)
						if abs(abs(a[0]-c[0])-math.pi/4) < 0.1:
							for d in angles3:
								if abs(abs(c[0]-d[0])-math.pi/2) < 0.1:
									return True
		return False
		
	except TypeError:
		return False

def box_arrow(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	minLineLength = 100
	maxLineGap = 5
	lines = cv2.HoughLinesP(edges,1,np.pi/180,55,minLineLength,maxLineGap)
	x = []
	y = []
	for x1,y1,x2,y2 in lines[0]:
		x.append(x1)
		x.append(x2)
		y.append(y1)
		y.append(y2)
	l_x = min(x)-4
	r_x = max(x)+4
	t_y = min(y)-4
	b_y = max(y)+4
	cv2.rectangle(img, (l_x,t_y), (r_x,b_y), (0,0,255),2)

	return img
	
# detect circles in an image. Returns true if a circle satisfying 
# certain parameters/thresholds is found. Returns false otherwise.
def detect_circle(img):
	gray = RGB_to_gray(img)
	try:
		circles = cv2.HoughCircles(gray,cv.CV_HOUGH_GRADIENT,2,50,
				                    param1=50,param2=215,minRadius=5,maxRadius=250)
		circles2 = circles[0]
		for circle in circles2:
			x = int(circle[0])
			y = int(circle[1])
			rad = int(circle[2])
			u_l = (x-rad-4,y-rad-4)
			b_r = (x+rad+4,y+rad+4)
			cv2.rectangle(img, u_l, b_r, (255,0,0),2)
		return True, img
	except TypeError:
		return False, img

# Question 2
for i in range(1,24):
	a = imread("Test" + str(i) + ".png")
	b = RGB_to_gray(a)
	x = detect_arrow(a)
	if x:
		a = box_arrow(a)
	y, a = detect_circle(a)
	print "image", i, x, y
	imsave(a, "result" + str(i) + ".jpg")

