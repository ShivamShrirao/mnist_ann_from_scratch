#!/usr/bin/env python3
import numpy as np

sd=np.random.randint(500)
print(sd)
np.random.seed(sd)

class neural_net:
	def __init__(self, nrons):
		self.nrons = nrons
		self.weights=[]
		self.bias=[]
		for i in range(len(self.nrons)-1):
			self.weights.append(2*np.random.rand(self.nrons[i],self.nrons[i+1])-1)
			self.bias.append(2*np.random.rand(1,self.nrons[i+1])-1)

	def __str__(self):
		return str(self.__dict__)

	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))

	def sigmoid_der(self,x):
		return x * (1 - x)

	def feed_forward(self, X):			# X = [i1, i2, i3, i4, i5]
		self.X = X.reshape(1,self.nrons[0])
		self.a = [self.X]
		for i in range(len(self.nrons)-1):
			z = (np.dot(self.a[i] ,self.weights[i])+self.bias[i])
			self.a.append(self.sigmoid(z))
		self.out = self.a[-1]
		return self.out

	def backprop(self, y):
		self.y = y
		d_c_z2	= (2*(self.y-self.out)*self.sigmoid_der(self.out))
		d_c_w2	= np.dot(d_c_z2.T,self.a[2])
		d_c_b2	= d_c_z2.sum()
		d_c_z	= (np.dot(d_c_z2,self.w2)*self.sigmoid_der(self.a))
		d_c_w1	= np.dot(self.X.T, d_c_z)
		d_c_b1	= d_c_z.sum()
		self.w1+=d_c_w1
		self.b1+=d_c_b1
		self.w2+=d_c_w2
		self.b2+=d_c_b2
