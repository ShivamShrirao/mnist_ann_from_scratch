#!/usr/bin/env python3
import numpy as np

sd=470#np.random.randint(500)
print(sd)
np.random.seed(sd)	#470

class neural_net:
	def __init__(self, nrons):
		self.nrons = nrons
		self.weights=[]
		self.bias=[]
		self.learning_rate=0.1
		for i in range(len(self.nrons)-1):
			self.weights.append(2*np.random.rand(self.nrons[i],self.nrons[i+1])-1)
			self.bias.append(2*np.random.rand(1,self.nrons[i+1])-1)

	def __str__(self):
		return str(self.__dict__)

	def sigmoid(self,x):
		return 1.0/(1+np.exp(-x))

	def sigmoid_der(self,x,y):
		return x * (1 - x)

	def relu(self,x):
		return x*(x>0)

	def relu_der(self,x,y):
		return (y > 0)*1

	def softmax(self,x):
		exps = np.exp(x)
		# exps = np.exp(x-np.max(x))
		return exps/np.sum(exps)

	def soft_der(self,x,y):
		return x * (1 - x)

	def activations(self,func):
		self.func=func							# ['sigmoid','relu','softmax']
		self.activate=[]
		self.act_der=[]
		for i in range(len(func)):
			if func[i]=='relu':
				self.activate.append(self.relu)
				self.act_der.append(self.relu_der)
			elif func[i]=='softmax':
				self.activate.append(self.softmax)
				self.act_der.append(self.soft_der)
			elif func[i]=='sigmoid':
				self.activate.append(self.sigmoid)
				self.act_der.append(self.sigmoid_der)

	def feed_forward(self, X):
		self.X = X.reshape(1,self.nrons[0])
		self.z = []
		self.a = [self.X]						# a0(1,784)
		for i in range(len(self.nrons)-1):
			self.z.append(np.dot(self.a[i] ,self.weights[i])+self.bias[i])	# w0(784,20) w1(20,20) w2(20,10)
			self.a.append(self.activate[i](self.z[-1]))		# a1(1,20) a2(1,20) b
		return self.a[-1][0]					# a3(1,10)

	def backprop(self, y):
		self.y = y 								# (1,10)
		d_c_a = 2*(self.y-self.a[-1])
		for i in range((len(self.nrons)-2), -1, -1):
			d_c_b = d_c_a*self.act_der[i](self.a[i+1],self.z[i])
			d_c_w = np.dot(self.a[i].T, d_c_b)
			d_c_a = np.dot(d_c_b, self.weights[i].T)
			self.weights[i]+=(d_c_w*self.learning_rate)
			self.bias[i]+=(d_c_b*self.learning_rate)