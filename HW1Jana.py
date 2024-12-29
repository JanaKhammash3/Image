import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

image = cv2.imread('C:/Users/hp/Desktop/dog.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#show the histogram of the image
im = cv2.imread('C:/Users/hp/Desktop/gray.jpg')
# calculate mean value from RGB channels and flatten to 1D array
vals = im.mean(axis=2).flatten()
# calculate histogram
counts, bins = np.histogram(vals, range(257))
# plot histogram centered on values 0..255
plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
plt.xlim([-0.5, 255.5])
plt.show()

def dir(skew):
	if skew > 0:
		return "right"
	elif skew < 0:
		return "left"
	else:
		return "normally"
print("Histogram features: ")
print("mean: " + str(np.mean(counts)) + ", mode = " + str(stats.mode(counts)[0][0]) + ", median = " + str(np.median(counts)) + ", skew = " + str(stats.skew(counts)) + ", direction = " + dir(stats.skew(counts)) + " skewed")

img = cv2.imread('C:/Users/hp/Desktop/gray.jpg',0)
# find frequency of pixels in range 0-255
histr = cv2.calcHist([img],[0],None,[256],[0,256])
# show the plotting graph of an image
plt.plot(histr)
plt.show()

#contrast
img = cv2.imread('C:/Users/hp/Desktop/gray.jpg')
original = img.copy()
xp = [0, 64, 128, 192, 255]
fp = [0, 16, 128, 240, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')
img = cv2.LUT(img, table)
cv2.imshow("original", original)
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()