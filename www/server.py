#!/usr/bin/env python3

from flask import Flask, request, render_template
import numpy as np
import pickle
import matplotlib.pyplot as plt
from json import dumps
from sys import path
path.insert(0,'..')

with open('../trained.dump','rb') as f:
	ann=pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/submit',methods = ['POST'])
def submit():
	img = request.form['input']
	frame=np.fromiter(img.split(','), np.int)
	frame=(frame/255)
	out=ann.feed_forward(frame)
	ans=out.argmax()

	# plt.text(19, 1,'Prediction: {}'.format(ans))
	# plt.text(17, 2,'Confidence: {}'.format(str(round(out[ans]*100,2))+"%"))
	# plt.imshow(frame.reshape(28,28), cmap='Greys')
	# plt.show()

	jj = {'prediction': int(ans), 'confidence' : round(out[ans]*100,2)}
	return dumps(jj)

if __name__ == '__main__':
	app.run(debug = True)