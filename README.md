# Game Review Sentiment Analyzer & Summariser

my project sumarises reviews by:

- classifying thhe review with a sentiment (positive or negative)
- generating a short summary using the most informative statement

## How it works

### Training

- Reviews are loaded from the NLTK `movie_reviews` dataset
- Data is split into training and test sets (80/20, stratified)
- Text is converted into numerical features using TF-IDF
- A Logistic Regression model from sklearn learns sentiment patterns
- trained pipeline is saved

### Analysis

 - saved model is loaded
 - User input is classified as positive or negative using trained model
 - A confidence score is calculated using class probabilities (from training test)
 - The review is summarised using extractive sentence scoring  (2 most informative senances of review combined)

## Performance

Accuracy: ~82% on test data

## Technologys used

- Python 3 (programming language)
- scikit-learn (helper classes and functions in training ai)
- NLTK (movie review dataset)
- TF-IDF (converts text into numerical weights)
- Logistic Regression (learns sentiment of weights through iteration)
- joblib (used to save and load the model)
