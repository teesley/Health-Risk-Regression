import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_nav import Nav
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/") 
def Home():
    return render_template("home.html")

@app.route("/data")
def dataDet():
    return render_template("data_details.html")

@app.route("/model")
def modelDet():
    return render_template("model_details.html")

@app.route("/predict", methods = ["POST"])
def predict():
    x_features = [float(x) for x in request.form.values()]
    features = [np.array(x_features)]
    prediction = model.predict(features)
    text = "Maternal Health risk is {}".format(prediction)
    text = re.sub(r'[\[\]\']', '', text)
    return render_template("home.html", prediction_text = text)

if __name__ == '__main__':
    app.run(debug=True, port=5000)      