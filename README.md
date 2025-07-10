# 📧 Email Spam Detection using Naive Bayes and SVM

## 🚀 Overview

This project focuses on building a machine learning-based spam classifier to detect whether an email (or SMS) is spam or not. We implemented and compared two algorithms — **Multinomial Naive Bayes** and **Support Vector Machine (SVM)** — to find the best performer in terms of accuracy, precision, recall, and F1-score.

---

## 🎯 Objective

Automatically classify emails as **spam** or **ham** (not spam) using natural language processing (NLP) and machine learning. The goal is to build a lightweight, high-accuracy model that can be integrated into real-world applications.

---

## 🧾 Dataset

- **Name:** SMS Spam Collection Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Records:** ~5k+ messages  
- **Labels:**  
  - `ham`: legitimate (non-spam) messages  
  - `spam`: unwanted promotional/scam messages  

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:**  
  - `scikit-learn`  
  - `nltk`  
  - `pandas`, `numpy`  
  - `matplotlib`, `seaborn`  
  - `joblib` (for model saving)

---

## 🔧 Preprocessing Pipeline

1. Lowercase all text
2. Remove numbers and punctuation
3. Tokenize text
4. Remove stopwords using NLTK
5. Apply stemming using PorterStemmer
6. Convert to vectors using **TF-IDF**

---

## 🤖 ML Models Used

- ✅ **Multinomial Naive Bayes** – good baseline for NLP
- ✅ **Linear Support Vector Machine (SVM)** – often outperforms NB in precision

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

---

## 📈 Results

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Naive Bayes   | ~97%     | ~95%      | ~94%   | ~94%     |
| SVM           | ~98%     | ~97%      | ~96%   | ~96%     |

*SVM slightly outperforms Naive Bayes across most metrics.*

---

## 🧪 How to Run the Project

### 🔁 1. Clone the Repository

git clone https://github.com/yourusername/spam-detector.git
cd spam-detector

### 2. Install Dependencies
pip install -r requirements.txt

### 🧠 3. Run the Notebook
jupyter notebook spam_detection.ipynb

### 💾 Optional: Save Trained Model
import joblib
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

### 🧠 Future Improvements
✅ Build a web UI using Streamlit

✅ Add n-gram tuning (bi-grams, tri-grams)

✅ Hyperparameter tuning with GridSearchCV

✅ Real-time deployment using Flask API

✅ Try deep learning models (LSTM, BERT)

### 🧑‍💻 Author
Akshara S.
Machine Learning and AI Intern | AI/ML Enthusiast
🔗 [LinkedIn](https://www.linkedin.com/in/akshara-s01/)
