# ✈️ Airline Passenger Satisfaction Classifier

A machine learning project that predicts whether airline passengers are **satisfied or dissatisfied** based on flight experience data.

---

## 📌 About

This repository contains a passenger satisfaction prediction system built using multiple machine learning algorithms. The project focuses on analyzing airline passenger data, training classification models, and comparing their performance to determine the most effective approach.

---

## 🚀 Features

* Multiple classification models for prediction
* Data preprocessing and feature scaling
* Model evaluation with standard metrics
* Visualization of model performance
* Simple and interactive Streamlit interface

---

## 🧠 Models Used

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Neural Network
* Gaussian Naive Bayes

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Abubini/Airline-Passenger-Satisfaction-classifier.git
cd Airline-Passenger-Satisfaction-classifier
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
streamlit run main.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

## 📁 Project Structure

```
Airline-Passenger-Satisfaction-classifier/
├── main.py
├── requirements.txt
├── src/
├── data/
├── models/
├── results/
├── visuals/
└── README.md
```

---

## 📊 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

**Best Performing Model:** Random Forest

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Ensemble learning techniques
* REST API deployment
* Deep learning-based models

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* Kaggle Airline Passenger Satisfaction Dataset
* Scikit-learn and Streamlit communities
