import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import dlib
import csv
import exercise1Lib
import numpy as np
from skimage import io
from skimage import transform
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Load exampleImg.jpg, using skimage.io.imread()
img = io.imread('exampleImg.jpg')

# 2. Initializing face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 3. Detecting face, return rectangles, each rectangle corresponds to one face.
# You need to fill the missing argument of this function
dets = detector(img, 1)

shape = 0
# Extracting the shape of the face in the first rectangle (using the first element of the rectangles variable
for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)
    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                              shape.part(1)))

    # Extract facial landmarks from shape by calling the shape2points() function.
    keypoints = exercise1Lib.shape2points(shape)

    # 1. Load the landmark position of the standard face model
    standardModel = np.zeros((68, 2))
    tmp = 0
    with open('mean.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            standardModel[tmp, 0] = float(row[0])
            standardModel[tmp, 1] = float(row[1])
            tmp += 1
    standardModel = standardModel * 500

    # 2. Calculating the transorfmation between the two set of keypoints
    # 2.1 Instantiating a PolynomialTransform() transform function
    tform = transform.PolynomialTransform()

    # 2.2 Calculating the transformation by calling the estimate() method.
    #     You do not need to retuern any value after calling this methods,
    #     because the transformation parameter is store in the object you instantiated after calling this methods.
    tform.estimate(standardModel, keypoints)

    # 3. Warping your example image using the transform.warp() function
    warped = transform.warp(img, tform)

    # 4. Crop the face from registered image using the provided cropFace function.
    cropedExampleFace = exercise1Lib.cropFace(warped, standardModel)

    # 5. Croping the face from the example image using detected landmarks
    cropedExampleFace2 = exercise1Lib.cropFace(img, keypoints)

    # Constructing figure with 2x3 subplots
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))

    # subplot [0,0]: show the original example image
    ax[0, 0].imshow(img)

    # Placing detected landmarks on subplot [0,0], we provide an exmaple to do this.
    for pointIte in range(len(keypoints)):
        # Create a Rectangle patch
        rect = patches.Rectangle((keypoints[pointIte][0] - 1, keypoints[pointIte][1] - 1),
                                 3, 3, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax[0, 0].add_patch(rect)

    # subplot [1,0]: show the face cropped from the example image.
    ax[1, 0].imshow(cropedExampleFace2, 'gray')

    # subplot [2,0]: show the histogram of the face cropped from the example image.
    Hist_croppedExampleImg, _ = np.histogram(
        img_as_ubyte(cropedExampleFace2).ravel(), bins=256)
    ax[2, 0].plot(Hist_croppedExampleImg)

    # subplot [0,1]: show the registered image
    ax[0, 1].imshow(warped)

    # place the model landmarks on the registered image
    for pointIte in range(len(standardModel)):
        # Create a Rectangle patch
        rect = patches.Rectangle((standardModel[pointIte][0] - 1, standardModel[pointIte][1] - 1),
                                 3, 3, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax[0, 1].add_patch(rect)

    # subplot [1,1]: show the face cropped from the registered image
    ax[1, 1].imshow(cropedExampleFace, 'gray')

    # subplot [2,1]: show the histogram of the face cropped from the registered image.
    Hist_croppedExampleImg, _ = np.histogram(
        img_as_ubyte(cropedExampleFace).ravel(), bins=256)
    ax[2, 1].plot(Hist_croppedExampleImg)

    fig.tight_layout()
    plt.show()
