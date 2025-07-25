from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained pipeline (vectorizer + model)
model = joblib.load('sentiment_model.pkl')

@app.route('/')
def home():
    return "✅ Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': '❌ Please provide text in JSON like { "text": "your input" }'}), 400

    input_text = data['text']
    prediction = model.predict([input_text])[0]

    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    predicted_label = label_map.get(prediction, "Unknown")

    return jsonify({
        'input': input_text,
        'sentiment': predicted_label
    })

if __name__ == '__main__':
    app.run(debug=True)
