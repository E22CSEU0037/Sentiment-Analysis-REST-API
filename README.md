# 🧠 Sentiment Analysis REST API

This is a production-ready machine learning API that predicts the sentiment (Positive, Neutral, Negative) of any given text input. The model is trained on real-world political tweets and exposed as a REST API using Flask. Deployed on [Render](https://render.com), the API is accessible publicly and ready for frontend or third-party integration.

---

## 📌 Features

- 💬 Predict sentiment of any text input
- 🧠 Trained using Scikit-learn + TF-IDF vectorizer
- 🚀 Deployed on Render with public API access
- 🧪 Tested with Postman and `curl`
- 🔁 Supports integration with any frontend (CORS ready)

---

## 🛠️ Tech Stack

- **Model:** Logistic Regression (Scikit-learn)
- **Vectorizer:** TF-IDF (max_features=5000)
- **API Framework:** Flask
- **Deployment:** Render
- **Testing Tools:** Postman, `curl`

---

## 🚀 Live API

🔗 **Base URL**: `https://sentiment-analysis-rest-api.onrender.com`

---

## 📖 API Documentation

### `GET /`
Health check route to confirm the API is live.

**Response:**
```text
✅ Sentiment Analysis API is running!
````

---

### `POST /predict`

Takes a text input and returns its sentiment classification.

#### 🔸 Request:

* **Method**: `POST`
* **Content-Type**: `application/json`
* **Body**:

```json
{
  "text": "I love this app!"
}
```

#### 🔹 Response:

```json
{
  "input": "I love this app!",
  "sentiment": "Positive"
}
```

#### 🔴 Error Response (Missing Text):

```json
{
  "error": "❌ Please provide text in JSON like { \"text\": \"your input\" }"
}
```

---

## 🧪 Testing the API

### ▶️ Using `curl`:

```bash
curl -X POST https://sentiment-analysis-rest-api.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d "{\"text\":\"Modi is a great leader!\"}"
```

---

### ▶️ Using Postman:

* Method: `POST`
* URL: `https://sentiment-analysis-rest-api.onrender.com/predict`
* Body → `raw` → `JSON`:

```json
{
  "text": "This is awful!"
}
```

---

## 🧠 Model Training Summary

* Dataset: Real-world Twitter sentiment data
* Labels:

  * `1` = Positive
  * `0` = Neutral
  * `-1` = Negative
* Vectorization: TF-IDF (stop words removed, max features = 5000)
* Classifier: Logistic Regression (`max_iter=1000`)
* Accuracy: \~ (Insert if evaluated)

---

## 📂 Project Structure

```
sentiment-api/
├── app.py                  # Flask app (API)
├── train_model.py          # Model training & saving
├── sentiment_model.pkl     # Saved pipeline (vectorizer + model)
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
└── README.md               # Project documentation
```

---

## 💡 Future Improvements

* Add frontend UI (React / HTML)
* Add logging and usage analytics
* Connect to a database for saving predictions
* Switch to more advanced models (BERT, LSTM)

---

## 🧑‍💻 Author

Made with 💛 by [GEETIKA SINGH](https://github.com/E22CSEU0037)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).


