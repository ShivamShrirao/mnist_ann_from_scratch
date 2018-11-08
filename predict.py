#!/usr/bin/env python3

import pickle
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

print("Loading dataset....")
mndata = MNIST('../mnist_dataset')
X_train, y_train = mndata.load_training()
X_train = mndata.process_images_to_numpy(X_train)/255
X_test, y_test = mndata.load_testing()
X_test = mndata.process_images_to_numpy(X_test)/255

with open('trained.dump','rb') as f:
	cnn=pickle.load(f)

while True:
	ng=np.random.randint(10000)
	out=cnn.feed_forward(X_test[ng])
	ans=out.argmax()
	print("I think number is:",ans)
	print("Confidence:",str(out[ans]*100)[:5],"%")
	y = np.zeros(10)
	y[y_test[ng]] = 1
	print("Correct answer is:",y_test[ng])
	print("Cost:",((y-out)**2).sum())

	plt.imshow(X_test[ng].reshape(28,28), cmap='Greys')
	plt.show()