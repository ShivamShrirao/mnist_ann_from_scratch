#!/usr/bin/env python3

import nnet
import pickle
import numpy as np
from mnist import MNIST
from time import time

print("Loading dataset....")
mndata = MNIST('../mnist_dataset')
X_train, y_train = mndata.load_training()
X_train = mndata.process_images_to_numpy(X_train)/255
X_test, y_test = mndata.load_testing()
X_test = mndata.process_images_to_numpy(X_test)/255

cnn=nnet.neural_net(nrons=[784,50,30,10])
cnn.learning_rate=0.1
# cnn.activations(func=['sigmoid','relu','softmax'])
y_inp=np.zeros((len(y_train),10))
for i in range(len(y_train)):
	y_inp[i][y_train[i]]=1

print("Training....")

t=time()
for i in range(len(X_train)):
	out=cnn.feed_forward(X_train[i])
	cnn.backprop(y_inp[i])
	if not i%1000:
		print('\rProgress:',str(i/600)[:5].ljust(5),'%',end='')

print()
print("Time:",(time()-t))
with open('trained.dump','wb') as f:
	pickle.dump(cnn,f)

print("Calculating accuracy....")
correct=0
for i in range(len(X_train)):
	out=cnn.feed_forward(X_train[i])
	ans=out.argmax()
	# cnn.backprop(y_inp[i])
	cor=y_train[i]
	if ans == cor:
		correct+=1
	if not i%1000:
		print('\rProgress:',str(i/600)[:5].ljust(5),'%',end='')

print()
print("Training accuracy:",(correct*100/len(y_train)),'%')

correct=0
for i in range(len(X_test)):
	out=cnn.feed_forward(X_test[i])
	ans=out.argmax()
	cor=y_test[i]
	if ans == cor:
		correct+=1

print("Testing accuracy:",(correct*100/len(y_test)),'%')