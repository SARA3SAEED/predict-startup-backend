import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

appli = Flask(__name__)
CORS(appli) 
app = appli

# Load the model and scaler
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

def encode_state(state):
    if state == "Florida":
        return [1, 0] 
    elif state == "New York":
        return [0, 1]  
    else:  
        return [0, 0]  

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the prediction API. Use POST to make predictions."})

@app.route('/', methods=['POST'])
def prediction_fun():
    data = request.get_json()
    rd_spend = float(data['rdSpend'])
    admin = float(data['admin'])
    state = data['state']

    state_encoded = encode_state(state)
    features = np.array([rd_spend, admin] + state_encoded).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = round(model.predict(features_scaled)[0], 2)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081, debug=True)