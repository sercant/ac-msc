# Initialzing your SVM classifiers.
from sklearn import svm
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Load data
exerciseData = sio.loadmat('lab2_data.mat')

#    1.1 Load 'training_data_proso'
training_data_proso = exerciseData['training_data_proso']

#    1.2 Load 'training_data_mfcc'
training_data_mfcc = exerciseData['training_data_mfcc']


#    1.3 Load 'training_class'
training_class = exerciseData['training_class'].reshape(-1)


# 2. Train your classifier using the prodosic data
#    2.1 Initialize your svm classifer
clf_proso = svm.SVC(kernel='poly', degree=3)

#    2.2 Train you classifier
clf_proso.fit(training_data_proso, training_class)

# 3. Train your classifer using the mfcc data
#    3.1 Initialize your svm classifer
clf_mfcc = svm.SVC(kernel='poly', degree=3)

#    3.2 Train you classifier
clf_mfcc.fit(training_data_mfcc, training_class)

# 4. Load testing data
testing_data_mfcc = exerciseData['testing_data_mfcc']
testing_data_proso = exerciseData['testing_data_proso']
testing_class = exerciseData['testing_class'].reshape(-1)

prediction_proso_training = clf_proso.predict(training_data_proso)
prediction_proso_testing = clf_proso.predict(testing_data_proso)
prediction_mfcc_training = clf_mfcc.predict(training_data_mfcc)
prediction_mfcc_testing = clf_mfcc.predict(testing_data_mfcc)

# 5. Calculate the average classification performances for the training data

accuracy_training_proso = accuracy_score(
    training_class, prediction_proso_training, normalize=True)
accuracy_training_mfcc = accuracy_score(
    training_class, prediction_mfcc_training, normalize=True)


# 6. Calculate the average classification performance for the testing data
accuracy_testing_proso = accuracy_score(
    testing_class, prediction_proso_testing, normalize=True)
accuracy_testing_mfcc = accuracy_score(
    testing_class, prediction_mfcc_testing, normalize=True)


# 7. Print the four accuracies.
print(
    '--- Four accuracies ---\n',
    'accuracy_training_proso: ', accuracy_training_proso, '\n',
    'accuracy_training_mfcc: ', accuracy_training_mfcc, '\n',
    'accuracy_testing_proso: ', accuracy_testing_proso, '\n',
    'accuracy_testing_mfcc: ', accuracy_testing_mfcc, '\n',
)

# 8. Visulize the confusion matrix
conf_proso_training = confusion_matrix(
    training_class, prediction_proso_training)
conf_mfcc_training = confusion_matrix(training_class, prediction_mfcc_training)

conf_proso_testing = confusion_matrix(testing_class, prediction_proso_testing)
conf_mfcc_testing = confusion_matrix(testing_class, prediction_mfcc_testing)

print(
    '--- Confusion matrices ---\n',
    'conf_proso_training\n', conf_proso_training, '\n',
    'conf_mfcc_training\n', conf_mfcc_training, '\n',
    'conf_proso_testing\n', conf_proso_testing, '\n',
    'conf_mfcc_testing\n', conf_mfcc_testing, '\n'
)
