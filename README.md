# 🕵️‍♂️ Instagram Fake Profile Detection - A Machine Learning Approach 🕵️‍♂️

Instagram Fake Profile Detection is a web-based project that leverages machine learning and deep learning models to detect fake Instagram accounts using various profile-based features. The current implementation is backend-focused with an ML-powered prediction engine, while frontend features like login and homepage are part of future development.

---

## 🌟 Features

- 🧠 **Machine Learning Models**: SVM, Random Forest, ANN
- 🤖 **Deep Learning Models**: GRU, Hybrid RF-GRU
- 📊 **Live Prediction**: Upload or input user profile data to classify accounts as fake or real.
- 🧪 **Model Comparison**: Accuracy and performance comparison across ML and DL models.
- 🔒 **Future Work**: User login system and responsive dashboard.

---
## 📂 Project Structure

The project is structured as follows:

```
instagram-fake-profile-detection/
├── models/                      # Trained ML/DL model files (.pkl and .h5)
├── app/                         # Flask app with main backend logic
│   ├── main.py                  # Core API routes
│   ├── utils.py                 # ML/DL model functions
├── dataset/
│   └── instagram.csv            # Dataset with user profile features
├── requirements.txt             # Python dependencies
└── README.md                    # This logbook
```

---

## ⚙️ Requirements

- 🐍 Python 3.x
- 🖼️ Visual Studio Code
- 🧪 Flask
- 📦 Scikit-learn, Keras, TensorFlow
- 📉 Matplotlib, Pandas, Seaborn, NumPy

 ---

## 🚀 Installation & Setup

### 🖼️Backend Setup
1. Clone the repository:

  ```bash
  git clone https://github.com/omchaudhari01/instagram-fake-profile-detection.git
  ```

2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. Run the Flask server
  ```bash
   python app/main.py
  ```

 ---

## 📌 Core Modules
#### 1. **Data Import & Processing**
   
- Load the Instagram dataset, encode and normalize the features.
- View correlation using heatmaps.


#### 2. **Model Training**

- Random Forest, SVM, and ANN models trained using scikit-learn.
- Save trained models using pickle.

#### 3. **Prediction Functions**

- Predict using SVM, Random Forest, ANN models.
- Predict using GRU, LSTM, and Hybrid models using Keras.
- live_prediction() and live_prediction_dl() handle real-time inference.

#### 4. **Model Loading**

- Models are pre-trained and loaded from disk using pickle and Keras.

 ---
## 🌐 Planned API Endpoints

- `POST /predict_ml`: ML model-based classification
- `POST /predict_dl`: DL model-based classification
- `GET /health`: Check API health/status

---

## 🖥️ Future Work (Frontend)

- 🌐 Create a login and registration system.
- 📈 Develop a dashboard to upload data and view predictions.
- 📊 Visualization of model performance and prediction results.

---

## 📖 Summary

This project aims to tackle the increasing threat of fake Instagram accounts using advanced machine learning and deep learning techniques. It provides a foundation for real-time detection and can be extended into a fully-fledged web platform.

---

## ✍️ Author

- 👨‍💻 Created by [om chaudhari]
- 📅 2024-2025 | Department of Artificial Intelligence & Machine Learning
