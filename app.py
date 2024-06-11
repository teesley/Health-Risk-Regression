import numpy as np
from flask import Flask, render_template
from flask import request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    x_features = [int(x) for x in request.form.values()]
    features = [np.array(x_features)]
    prediction = model.predict(features)
    
    return render_template("index.html", prediction_text = "Maternal Health risk is {}".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)      