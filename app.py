

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model/calorie_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = 1 if request.form['gender'] == 'female' else 0
    features = [
        gender,
        float(request.form['age']),
        float(request.form['height']),
        float(request.form['weight']),
        float(request.form['duration']),
        float(request.form['heart_rate']),
        float(request.form['body_temp'])
    ]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction=f"Calories Burned: {prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)