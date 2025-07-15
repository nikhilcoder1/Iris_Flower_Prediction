from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [float(x) for x in request.form.values()]
    data = np.array([features])
    prediction = model.predict(data)

    # Map class index to actual name
    labels = ['Setosa', 'Versicolor', 'Virginica']
    result = labels[prediction[0]]

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
