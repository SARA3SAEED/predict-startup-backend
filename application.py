import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

model=pickle.load(open('models/model.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route('/', methods=['POST','GET'])
def prediction_fun():
    if request.method == 'POST':
        rd_spend = float(request.form['rdSpend'])
        admin = float(request.form['admin'])
        state = request.form['state']

        state_encoded = encode_state(state)  
        features = np.array([rd_spend, admin] + state_encoded).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = round(model.predict(features_scaled)[0], 2)

        return render_template('home.html', prediction=prediction)
    else:
        return render_template('home.html')

def encode_state(state):
    if state == "Florida":
        return [1, 0] 
    elif state == "New York":
        return [0, 1]  
    else:  
        return [0, 0]  



if __name__ == '__main__':
    app.run(host="0.0.0.0" , port=8080)