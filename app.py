from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/model_heart.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari form (harus sesuai urutan fitur saat training)
    input_data = [float(x) for x in request.form.values()]
    prediction = model.predict([input_data])[0]

    hasil = "Pasien berisiko penyakit jantung." if prediction == 1 else "Pasien tidak berisiko penyakit jantung."
    return render_template('result.html', prediction_text=hasil)

if __name__ == '__main__':
    app.run(debug=True)
