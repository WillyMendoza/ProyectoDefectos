# Importando librerias
import numpy as np
import cv2

# Lectura de la imagen
img = cv2.imread('plomo.jpg')

# Conversión a grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Inversion de imagen
inverted_image = np.invert(gray)

# Processamiento de imagen ( smoothing )
# Averaging
blur = cv2.blur(inverted_image,(3,3))

# Apply logarithmic transform
img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255

# Specify the data type
img_log = np.array(img_log,dtype=np.uint8)

# Image smoothing: bilateral filter
bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

# Canny Edge Detection
edges = cv2.Canny(bilateral,30,70)

# Morphological Closing Operator
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Create feature detecting method
#sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

#from google.colab.patches import cv2_imshow
#cv2_imshow(closing)

# Make featured Image
keypoints, descriptors = orb.detectAndCompute(closing, None)
featuredImg = cv2.drawKeypoints(closing, keypoints, None)

# Resultante
cv2.imshow('ORIGINAL', img)
cv2.imshow('DEFECTO', closing)
cv2.imshow('FEATURE', featuredImg)
cv2.waitKey()
 
