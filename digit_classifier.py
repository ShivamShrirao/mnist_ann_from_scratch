#!/usr/bin/env python3

import nnet
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

mndata = MNIST('../mnist_dataset')
X_train, y_train = mndata.load_training()
X_train = mndata.process_images_to_numpy(X_train)/255
X_test, y_test = mndata.load_testing()
X_test = mndata.process_images_to_numpy(X_test)/255

# plt.imshow(X_train[660].reshape(28,28), cmap='Greys')
# plt.show()

cnn=nnet.neural_net(nrons=[784,20,20,10])
# cnn.activations(func=['sigmoid','relu','softmax'])
for i,j in zip(range(len(X_train)),y_train):
	y = np.zeros(10)
	cnn.feed_forward(X_train[i])
	y[y_train[j]] = 1
	cnn.backprop(y)
	if not i%100:
		print('\rProgress:',i/600,' %',end='')