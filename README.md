# Quora Question Pairs
Kaggle competition to identify duplicate questions: https://www.kaggle.com/c/quora-question-pairs

## Process
### #1 Cosine Similarity with TF-IDF with the default tokenizer
### #2 Cosine Similarity with TF-IDF with a custom tokenizer
Each question is tokenized into sentences. Each sentence is tokenized into words. Each word is lemmatized.
### #3 Add some basic features and use the default settings of:
1. scikit-learn's Decision Tree Classifier
2. scikit-learn's AdaBoost Classifier
3. scikit-learn's Logistic Regression
4. dmlc's XGBoost
### #4 Reduce TF-IDF to 100 components and add as features. Use the default settings of:
1. scikit-learn's Logistic Regression
2. dmlc's XGBoost
### #5 Neural Network with 3 hidden layers (215 > 512 > 128 > 64 > 1)
1. Each layer consists of: Batch Normalization, Dropout (50%), XW+b, ReLU
2. (Batch size, Learning rate) = (1000, 0.1)
