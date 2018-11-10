from flask import Flask, request, render_template
from base64 import urlsafe_b64decode
# from PIL import Image
import numpy as np
# from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/submit',methods = ['POST'])
def submit():
	if request.method == 'POST':
		img = request.form['input']
		# img = img.replace('data:image/png;base64,', '')
		# img = img.replace(' ', '+')
		# img = urlsafe_b64decode(img)
		# im_frame = Image.open(BytesIO(img))
		# np_frame = np.array(im_frame.getdata())
		np_frame=list(map(int,img.split(',')))
		print(len(np_frame))
		return 'Success'

if __name__ == '__main__':
	app.run(debug = True)