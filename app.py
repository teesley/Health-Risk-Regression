import numpy as np
from flask import Flask, render_template
from flask import request
import pickle

app = Flask(__name__)

model = pickle.load(open("regression.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    x_features = [int(x) for x in request.form.values()]
    features = [np.array(x_features)]
    prediction = model.predict(features)
    
    return render_template("index.html", prediction_text = "Your maternal health his is {}".format(prediction))

if __name__ == '__main__':
    app.run(port=3000, debug=True)      