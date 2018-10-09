import os
import sys
import dlib
from skimage import io, transform, color, img_as_ubyte
import sklearn
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.io as sio

print('--- START TASK 1 ---')

mdata = sio.loadmat('./Exercise3/lab3_data.mat')
# #facial expression training and testing data, training and testing class
training_data = mdata['training_data']
testing_data = mdata['testing_data']
training_class = mdata['training_class'].ravel()
testing_class = mdata['testing_class'].ravel()

# #audio training and testing data
training_data_proso = mdata['training_data_proso']
testing_data_proso = mdata['testing_data_proso']

# set ReducedDim for facial expression feature and audio feature, respectively.
reducedDim_v = 20
reducedDim_a = 15

# Extract subspace for facial expression feature though PCA
# set n_components
pca_v = PCA(n_components=reducedDim_v)
pca_v.fit(training_data)
# Transform training_data and testing data respectively
training_data_transformed = pca_v.transform(training_data)
testing_data_transformed = pca_v.transform(testing_data)

# Extract subspace for audio features though PCA
pca_a = PCA(n_components=reducedDim_a)
pca_a.fit(training_data_proso)
# Transform training_data and testing data respectively
training_data_proso_transformed = pca_a.transform(training_data_proso)
testing_data_proso_tansformed = pca_a.transform(testing_data_proso)


# Concatenate ‘video training_data’ and ‘audio training_data’ into a new feature ‘combined_trainingData’
sample_train = np.concatenate(
    (training_data_transformed, training_data_proso_transformed), axis=1)

# Concatenate ‘video testing_data’ and ‘audio testing_data2 into a new feature ‘combined_testingData’.
sample_test = np.concatenate(
    (testing_data_transformed, testing_data_proso_tansformed), axis=1)

# Train SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(sample_train, training_class)

# The prediction results of training data and testing data respectively
pred_train = clf.predict(sample_train)
pred_test = clf.predict(sample_test)

# Calculate and Print the training accuracy and testing accuracy.
print('training accuracy: {}'.format(accuracy_score(
    training_class, pred_train, normalize=True)))
print('testing accuracy: {}'.format(
    accuracy_score(testing_class, pred_test, normalize=True)))

print('confusion matrix training:\n{}'.format(
    confusion_matrix(training_class, pred_train)))
print('confusion matrix testing:\n{}'.format(
    confusion_matrix(testing_class, pred_test)))

print('--- END TASK 1 ---')

print('--- START TASK 2 ---')

# Use CCA to construct the Canonical Projective Vector (CPV)
cca = CCA()
cca.fit(training_data, training_data_proso)

# Construct Canonical Correlation Discriminant Features (CCDF) for training data and testing data
training_data_cca, training_data_proso_cca = cca.transform(
    training_data, training_data_proso)
testing_data_cca, testing_data_proso_cca = cca.transform(
    testing_data, testing_data_proso)

# Concatenate multiple feature for training data and testing data respectively
training_CCDF = np.concatenate(
    (training_data_cca, training_data_proso_cca), axis=1)
testing_CCDF = np.concatenate(
    (testing_data_cca, testing_data_proso_cca), axis=1)

# Train SVM classifier
clf_CCDF = svm.SVC(kernel='linear')
clf_CCDF.fit(training_CCDF, training_class)

# The prediction results of training data and testing data respectively
pred_train_CCDF = clf_CCDF.predict(training_CCDF)
pred_test_CCDF = clf_CCDF.predict(testing_CCDF)

# Calculate and Print the training accuracy and testing accuracy.
print('training accuracy: {}'.format(accuracy_score(
    training_class, pred_train_CCDF, normalize=True)))
print('testing accuracy: {}'.format(
    accuracy_score(testing_class, pred_test_CCDF, normalize=True)))

print('confusion matrix training:\n{}'.format(
    confusion_matrix(training_class, pred_train_CCDF)))
print('confusion matrix testing:\n{}'.format(
    confusion_matrix(testing_class, pred_test_CCDF)))

print('--- END TASK 2 ---')
