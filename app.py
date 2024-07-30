from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model (ensure the path is correct)
model_path = 'model/placement_model.pkl'
model = joblib.load(model_path)

# Function to preprocess user input
def preprocess_input(data):
    # Extract features
    age = int(data['age'])
    gender = 0 if data['gender'].lower() == 'male' else 1
    internships = int(data['internships'])
    cgpa = float(data['cgpa'])
    hostel = 1 if data['hostel'].lower() == 'yes' else 0
    backlogs = int(data['backlogs'])

    # Stream Encoding
    stream = data['stream'].lower()
    streams = ['civil', 'computer science', 'electrical', 'electronics and communication', 'information technology', 'mechanical']
    stream_encoded = [1 if stream == s else 0 for s in streams]

    # Create final feature vector
    features = [age, gender, internships, cgpa, hostel, backlogs] + stream_encoded
    return np.array(features).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        features = preprocess_input(data)
        
        # Predict using the model
        prediction = model.predict(features)
        placed_or_not = 'Placed' if prediction[0] == 1 else 'Not Placed'

        return render_template('predict.html', prediction=placed_or_not)
    
    return render_template('predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
