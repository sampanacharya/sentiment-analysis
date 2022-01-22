import numpy as np 
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
tf_idf = pickle.load(open("tf_idf.pkl", "rb"))

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
	text = request.form.values()
	text = " ".join([str(i) for i in text]).lower()
	text = tf_idf.transform([text]).toarray()
	prediction = model.predict(text)[0]
	return render_template("index.html", sentence=prediction)


if __name__ == "__main__":
	app.run(debug=True)