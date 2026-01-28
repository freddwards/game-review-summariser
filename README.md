# Game Review Sentiment Analyser & Summariser
- currently in progress -

my project sumarises reviews by:

- classifying the review with a sentiment (positive or negative)
- generating a short summary using the most informative statement

## How it works

### Training

- Reviews are loaded from the NLTK movie_reviews dataset
- Data is split into training and test sets (80/20, stratified)
- Text is converted into numerical features using TF-IDF
- A Logistic Regression model from sklearn learns sentiment patterns
- trained pipeline is saved

### Analysis

 - saved model is loaded
 - User input is classified as positive or negative using trained model
 - A confidence score is calculated using class probabilities (from training test)
 - The review is summarised using extractive sentence scoring  (2 most informative sentences of review combined)

## Performance

Accuracy: ~82% on test data

## Technologies used

- **Python 3** - programming language
- **scikit-learn** - machine learning models and utilities
- **NLTK** - movie review dataset and sentence tokenisation
- **TF-IDF** - converts text into numerical feature weights 
- **Logistic Regression** - learns sentiment from TF-IDF weights
- **joblib** - saves and loads the model

## Dataset

A large, well labelled public game review sentiment dataset was not available. the model was trained on the NLTK movie_reviews  but the project is tested on real game reviews to assess how well sentiment classification transfers across similar review domains. Accuracy is based on movie review test data.


