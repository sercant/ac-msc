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
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

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

    # 1. Define parameter to extract LBP feature in (8, 1) neighborhood:
    #    1.1 Please set the number of neighbour P = 8
    #    1.2 Please set the radius if circir R = 1.0
    #    2.3 Please set the method as 'nri_uniform'
    P = 8
    R = 1.0
    method = 'nri_uniform'

    # 2. Extracting the LBP face using local_binary_pattern()
    lbpImg = local_binary_pattern(cropedExampleFace, P, R, method)

    # 3. Calculate the histogram of the LBP face. Sum of vector can be calculated by calling numpy.sum()
    n_bins = int(lbpImg.max() + 1)

    hist_lbp, _ = np.histogram(
        lbpImg.ravel(), bins=n_bins)

    hist_lbp_norm = hist_lbp / float(np.sum(hist_lbp))

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    # 4. Visualize your result.
    #
    ax[0].imshow(lbpImg, 'gray')
    ax[1].stem(hist_lbp_norm)

    fig.tight_layout()
    plt.show()
