#!/usr/bin/env python3

import nnet
import pickle
import sys
import numpy as np
from mnist import MNIST
from time import time

print("Loading dataset....")
mndata = MNIST('../mnist_dataset')
X_train, y_train = mndata.load_training()
X_train = (mndata.process_images_to_numpy(X_train)/255).astype(np.float32)
X_test, y_test = mndata.load_testing()
X_test = (mndata.process_images_to_numpy(X_test)/255).astype(np.float32)

ann=nnet.neural_net(nrons=[784,128,32,10])
ann.learning_rate=0.01
ann.activations(func=['sigmoid','sigmoid','sigmoid'])
# ann.activations(func=['relu','relu','sigmoid'])

y_inp=np.zeros((len(y_train),10))
for i in range(len(y_train)):
	y_inp[i][y_train[i]]=1

print("Training....")
epoch=epochs=10
opt=True
total_t=time()
while epoch>0:
	print("\nEpoch:",(1+epochs-epoch),'/',epochs)
	epoch-=1
	correct=0
	t=time()
	for i in range(len(X_train)):
		out=ann.feed_forward(X_train[i])
		ans=out.argmax()
		ann.backprop(y_inp[i])
		cor=y_train[i]
		# sys.exit(0)
		if ans == cor:
			correct+=1
		if not i%1000:
			print('\rProgress:',round(i/600,2),'%',end='')
	print()
	if epoch<(0.3*epochs) and opt:
		ann.learning_rate/=10
		opt=False
	print("Accuracy:",(correct*100/len(y_train)),'%')
	print("Time:",(time()-t))
	correct=0
	for i in range(len(X_test)):
		out=ann.feed_forward(X_test[i])
		ans=out.argmax()
		cor=y_test[i]
		if ans == cor:
			correct+=1
	print("Testing accuracy:",(correct*100/len(y_test)),'%')
# sys.exit(0)

print("\nTotal time:",(time()-total_t),"sec")

with open('trained.dump','wb') as f:
	pickle.dump(ann,f)

print("Calculating accuracy....")

correct=0
for i in range(len(X_train)):
	out=ann.feed_forward(X_train[i])
	ans=out.argmax()
	cor=y_train[i]
	if ans == cor:
		correct+=1
	if not i%1000:
			print('\rProgress:',round(i/600,2),'%',end='')
print()
print("Training accuracy:",(correct*100/len(y_train)),'%')

correct=0
for i in range(len(X_test)):
	out=ann.feed_forward(X_test[i])
	ans=out.argmax()
	cor=y_test[i]
	if ans == cor:
		correct+=1

print("Testing accuracy:",(correct*100/len(y_test)),'%')