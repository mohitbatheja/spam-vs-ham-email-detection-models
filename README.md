# Spam vs Ham Email Detection using Multinomial Naive Bayes 📱📩

A Machine Learning model built with Python to classify SMS messages as either **Spam** or **NonSpam** (Ham). Using `scikit-learn`'s **Multinomial Naive Bayes** classifier and **CountVectorizer** feature extraction, this model accurately filters out spam messages based on textual content.

---

## 📌 Features

* **Data Cleaning:** Handles missing values and removes duplicate entries to ensure high-quality training data.
* **Text Processing:** Converts raw text messages into word frequency vectors using `CountVectorizer`.
* **Model Training:** Utilizes `MultinomialNB` (Multinomial Naive Bayes) for fast and effective text classification.
* **Performance Evaluation:** Evaluated using accuracy score, detailed classification metrics (precision, recall, F1-score), and side-by-side output comparisons.

---

## 📊 Dataset

The model processes a dataset (`Spam_SMS.csv`) containing 5,574 text messages categorized into two classes:

* **Spam:** Promotional or malicious messages.
* **NonSpam (Ham):** Regular user messages.

---

## 📈 Model Performance

The Multinomial Naive Bayes model achieves an outstanding **~98.16% Accuracy** on the test set.

### Classification Report

```text
              precision    recall  f1-score   support

     NonSpam       0.99      0.99      0.99       911
        spam       0.93      0.91      0.92       121

    accuracy                           0.98      1032
   macro avg       0.96      0.95      0.96      1032
weighted avg       0.98      0.98      0.98      1032

```

---

## 🛠️ Requirements & Dependencies

Make sure you have Python 3.x installed along with the required libraries:

```bash
pip install pandas scikit-learn

```

---


```


1. **Ensure Dataset Location:**
Place the dataset file `Spam_SMS.csv` in the root directory.
2. **Run the Jupyter Notebook:**
Launch Jupyter Notebook or Jupyter Lab to run the steps:
```bash
jupyter notebook


---

## ⚙️ Workflow Overview

1. **Data Preprocessing:**
* Read data with `pandas`.
* Check and drop duplicate entries.
* Rename target labels (`ham` $\rightarrow$ `NonSpam`).


2. **Train-Test Split:**
* Split data into **80% training** and **20% testing** sets (`test_size=0.2`, `random_state=42`).


3. **Feature Extraction:**
* Transform text messages into numerical token count vectors using `CountVectorizer`.


4. **Model Training & Prediction:**
* Train `MultinomialNB()` on vectorized training data.
* Predict classes for test messages.


5. **Evaluation:**
* Compute accuracy and output full classification statistics using `classification_report`.
