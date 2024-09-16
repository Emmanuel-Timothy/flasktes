from flask import Flask, render_template, request
import numpy as np
import pickle
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model_knn = pickle.load(open('knn.pkl', 'rb'))
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

@app.route('/')
def home():
   return render_template("home.html")

@app.route('/training')
def training():
   
   return render_template("training.html")


@app.route('/predict', methods=['POST'])
def predict():
   sepal_length = request.form['sepallength']
   sepal_width = request.form['sepalwidth']
   petal_length = request.form['petallength']
   petal_width = request.form['petalwidth']
   arr = np.array(([sepal_length,sepal_width,petal_length,petal_width]))
   pred = model_knn.predict(arr)
   return render_template("predict.html", nama_spesies = pred)

if __name__ == '__main__':
   app.run(debug=True)