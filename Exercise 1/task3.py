import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import scipy.io as sio
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Loading data using scipy.io.loadmat(), or sio.loadmat
# Alternatively, you can us h5py also, if you would like to
mdata = sio.loadmat('Task3_data.mat')


# Load 'training_data'
sample_train = mdata['training_data']


# Load 'testing_data'
sample_test = mdata['testing_data']


# Load 'training_class'
label_train = mdata['training_class']

# Load 'testing_class'
label_test = mdata['testing_class']

# 1. Initializing a SVM classifier, using linear kernel
clf = svm.SVC(kernel='linear', cache_size=5000)

# 2. using the classifier to fit your training data
clf.fit(sample_train, label_train.ravel())

# 1. Predicting you training data and testing data.
prediction_train = clf.predict(sample_train)
prediction = clf.predict(sample_test)

# 2. Calculating the accuracies of your prediction on training data and testing data, respectively.
#    2.1 calculate the accuracy when classifying the training data
accuracy_training = accuracy_score(
    label_train.ravel(), prediction_train, normalize=True)

#    2.2 calculate the accuracy when classifying the test data
accuracy_test = accuracy_score(label_test.ravel(), prediction, normalize=True)

print("accuracy_training: {} accuracy_test: {}".format(
    accuracy_training, accuracy_test))

# 3. Draw your confusion matrix
# 3. using sklearn.metrics.confusion_matrix
#    3.1 Calculate the confusion matrix when classifying the training data
conf_train = confusion_matrix(label_train.ravel(), prediction_train)

#    3.2 Calculate the confusion matrix when classifying the testing data
conf_test = confusion_matrix(label_test.ravel(), prediction)

print("conf_train \n{}\n\n conf_test\n{}".format(
    conf_train, conf_test))
