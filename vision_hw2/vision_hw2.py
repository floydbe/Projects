import Image
import numpy as np
from scipy import misc
import os

def imread(filename):
	return np.array(Image.open(filename).convert('L')),np.array(Image.open(filename))
	
def imsave(im, filename):
	misc.imsave(filename, im)

def get_imlist(path, filetype):
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(filetype)]

def imresize(im, size):
	im = Image.fromarray(np.uint8(im))
	return np.array(im.resize(size))

testA_im_loc = "test/"
testA_data_loc = "testA_data.txt"

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
			cropped12 = imresize(cropped, (12,12))
			imsave(cropped12, line[0] + '12.jpg')
			cropped24 = imresize(cropped, (24,24))
			imsave(cropped24, line[0] + '24.jpg')
			cropped48 = imresize(cropped, (48,48))
			imsave(cropped48, line[0] + '48.jpg')

