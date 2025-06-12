from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
app = Flask(__name__)

CORS(app)
# Load model
model = load_model('model_h5.h5')

# Inisialisasi LabelEncoder
le = LabelEncoder()
le.classes_ = np.array(['aural', 'kinesthetic', 'read/write', 'visual'])

# Mapping gaya belajar huruf ke angka
style_mapping = {'V': 3, 'A': 0, 'R': 2, 'K': 1}

# Fungsi prediksi
def predict_vark_style(response_list):
    if len(response_list) != 16:
        raise ValueError("Input harus terdiri dari 16 jawaban (V, A, R, K)")

    try:
        input_data = [style_mapping[char.upper()] for char in response_list]
    except KeyError:
        raise ValueError("Input hanya boleh mengandung huruf: V, A, R, K (case-insensitive)")

    input_array = np.array([input_data])
    predictions = model.predict(input_array)
    predicted_index = np.argmax(predictions)
    predicted_label = le.inverse_transform([predicted_index])[0]

    return predicted_label

# Endpoint /predict
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'responses' not in data:
        return jsonify({'error': 'JSON harus memiliki key "responses"'}), 400

    responses = data['responses']

    try:
        prediction = predict_vark_style(responses)
        return jsonify({'predicted_style': prediction})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

# Run server
if __name__ == '__main__':
    app.run(debug=True, port=5000)
