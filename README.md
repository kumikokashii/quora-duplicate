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
1. Each layer consists of Batch Normalization, Dropout (50%), XW+b, ReLU
2. (Batch size, Learning rate) = (1000, 0.1)
### [ Prep ] POS tagging of the training questions with the default POS tagger
### [ Prep ] Manhattan LSTM
### #6 Convolutions + LSTM Neural Network
1. Replace up to 5 words of each question with their synonym
2. Swap question 1 and question 2 with a 50% possibility
3. **Convolutions**: For each of window size (3, 5, 7, 11), perform a convolution with 100 output feature maps, then max pooling for each feature map. Add (4 window sizes) x (100 values) x (2 questions) = (800 features) to the fully-connected layer.
4. **Long short-term memory**: Use the last output vector from each question, and the similarity between the two. Add (50 values) x (2 questions) + (1 similarity) = (101 features) to the fully-connected layer.
5. **Fully-connected** with 2 hidden layers (1116 > 256 > 32 > 1): Each layer consists of Batch Normalization, Dropout (50%), XW+b, ReLU
6. (Batch size, Learning rate) = (1000, 0.1)
