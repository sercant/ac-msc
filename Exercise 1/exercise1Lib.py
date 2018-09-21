import dlib
from skimage import io
from skimage import transform
from skimage import color
from skimage import img_as_ubyte
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def shape2points(shape, dtype = 'int', pointNum = 68):
    
    coords = np.zeros((pointNum, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    return coords

def cropFace(rawImage, landMarks, \
             leftEyeInds = [36, 37, 38, 39, 40, 41], \
             rightEyeInds = [42, 43, 44, 45, 46, 47], \
             doNormalization = False):
    
    # should be gray image
    if len(rawImage.shape) == 3 and rawImage.shape[2] == 3:
        rawImage = color.rgb2gray(rawImage)
        
    # extract the left eye coordinate
    leftEye = landMarks[leftEyeInds, :].sum(0).astype('float')/len(leftEyeInds)
    # extract the right eye coordinate
    rightEye = landMarks[rightEyeInds, :].sum(0).astype('float')/len(rightEyeInds)
    
    # distance between two eyes
    distBetweenEyes = np.sqrt(sum((leftEye - rightEye)**2))
    
    x1 = leftEye[0]
    y1 = leftEye[1]
    x2 = rightEye[0]
    y2 = rightEye[1]

    sina= (y1-y2)/distBetweenEyes
    
    cosa = (x2-x1)/distBetweenEyes
    
    lefttopy = y1 + distBetweenEyes * 0.4 * sina -distBetweenEyes * 0.6 * cosa
    lefttopx = x1 - distBetweenEyes * 0.4 * cosa - distBetweenEyes * 0.6 * sina

    faceHeight = int(round(distBetweenEyes * 2.2))
    faceWidth = int(round(distBetweenEyes * 1.8))
    
    norm_face = np.zeros((faceHeight, faceWidth))

    [wi, hi] = rawImage.shape

    for h in range(0, faceHeight):
        starty = lefttopy + h * cosa
        startx = lefttopx + h * sina
        
        for w in range(0, faceWidth):
            if np.uint16(starty - w * sina) > wi:
                norm_face[h,w] = rawImage[np.uint16(wi), np.uint16(startx + w * cosa)]
                
            elif np.uint16(startx + w * cosa) > hi:
                norm_face[h,w] = rawImage[np.uint16(starty - w * sina), np.uint16(hi)]
                
            else:
                norm_face[h,w] = rawImage[np.uint16(starty - w * sina), np.uint16(startx + w * cosa)]

    
    if doNormalization == 1:
        norm_face = transform.resize(norm_face, (128,128))
        
    return norm_face