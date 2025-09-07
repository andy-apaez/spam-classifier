📩 Spam Email Classifier

A simple machine learning project that classifies SMS messages as spam or ham (not spam) using Python, scikit-learn, and Naive Bayes.

---
🚀 Features

* Preprocesses raw text messages

* Converts text into numerical features using Bag of Words

* Trains a Naive Bayes classifier

* Evaluates accuracy with test data

* Allows users to test custom messages

---
🛠️ Tech Stack

* Python 3

* Pandas – data handling

* scikit-learn – ML model & preprocessing

* NLTK (optional) – text cleaning/tokenization

---
📂 Dataset

This project uses the SMS Spam Collection Dataset, which you can download from Kaggle

*label* → spam or ham

*message* → SMS content

---
📊 Example Output

           Accuracy: 0.98
           precision    recall  f1-score   support
              
           0       0.99      0.98      0.98       965
           1       0.92      0.97      0.94       150

   
           accuracy                           0.98      1115
           macro avg       0.95      0.97      0.96      1115
           weighted avg       0.98      0.98      0.98      1115

