from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('building_height_model.pkl')

@app.route('/')
def home():
    return render_template('mlproject.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Soil Type': [request.form['soil']],
        'Moisture (%)': [float(request.form['moisture'])],
        'Clay (%)': [float(request.form['clay'])],
        'Sand (%)': [float(request.form['sand'])],
        'Silt (%)': [float(request.form['silt'])],
        'pH': [float(request.form['ph'])],
        'Bearing Capacity (kPa)': [float(request.form['bc'])]
    }
    df = pd.DataFrame(input_data)
    prediction = model.predict(df)[0]
    return render_template('mlproject.html', result=f"Predicted Max Building Height: {prediction:.2f} m")

if __name__ == '__main__':
    app.run(debug=True)
