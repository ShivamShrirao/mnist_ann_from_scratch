#!/usr/bin/env python3

import nnet
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from time import time

mndata = MNIST('../mnist_dataset')
X_train, y_train = mndata.load_training()
X_train = mndata.process_images_to_numpy(X_train)/255
X_test, y_test = mndata.load_testing()
X_test = mndata.process_images_to_numpy(X_test)/255

cnn=nnet.neural_net(nrons=[784,50,30,10])
# cnn.activations(func=['sigmoid','relu','softmax'])
y=np.zeros((len(y_train),10))
for i in range(len(y_train)):
	y[i][y_train[i]]=1

t=time()
for i in range(len(X_train)):
	out=cnn.feed_forward(X_train[i])
	cnn.backprop(y[i])
	if not i%1000:
		print('\rProgress:',str(i/600)[:5],'%',end='')

print()
print("Time:",(time()-t))

ng=np.random.randint(10000)
out=cnn.feed_forward(X_test[ng])
ans=out.argmax()
print("I think number is:",ans)
print("Confidence:",str(out[ans]*100)[:5],"%")
y = np.zeros(10)
y[y_test[ng]] = 1
print("Correct answer is:",y_test[ng])
print("Cost:",((y-out)**2).sum())

# plt.imshow(X_test[ng].reshape(28,28), cmap='Greys')
# plt.show()

correct=0
for i in range(len(X_train)):
	out=cnn.feed_forward(X_train[i])
	ans=out.argmax()
	cor=y_train[i]
	if ans == cor:
		correct+=1

print("Training accuracy:",(correct*100/len(y_train)),'%')

correct=0
for i in range(len(X_test)):
	out=cnn.feed_forward(X_test[i])
	ans=out.argmax()
	cor=y_test[i]
	if ans == cor:
		correct+=1

print("Testing accuracy:",(correct*100/len(y_test)),'%')