import joblib # used to save/load trained pipeline 
from pathlib import Path # helps work with file paths
from src.summarise import summarise_text # imports summarise function to generate summary

# constant for file path 
MODEL_PATH = Path("models/sentiment_pipeline.joblib")

# function that loads trained model
def load_model():
    if not MODEL_PATH.exists(): # checks if model exists
        raise FileNotFoundError( # tell user what to do if model not found
            "Model not found. Make sure model has been trained and saved using train.py before running"
        )
    return joblib.load(MODEL_PATH) # model found successfully, it is loaded and returned

#  takes text and returns a positive or negative sentiment with a confidence rating (0-1)
def predict_sentiment(model, text: str):
    label = model.predict([text])[0]  # makes text into a list then predicts a sentiment label

    probs = model.predict_proba([text])[0] # probability scores for each class
    classes = list(model.classes_) # class labels (pos and neg) in the same order as probs
    confidence = float(probs[classes.index(label)]) # probability of predicted class (model confidence)

    return label, confidence

# an input helper so the user can paste in reviews
def read_multiline_input():
    print("Paste a review. Press ENTER on an empty line to finish:\n") # instruction for user
    lines = [] # list to store inputted lines
    while True: # keeps reading until user inputs empty line
        line = input()
        if line.strip() == "" and lines:
            break
        lines.append(line) # adds line to lines list
    return "\n".join(lines).strip() # returns all lines in one string seperated by new line with removed whitespace


def main():
    model = load_model() # loads model
    review = read_multiline_input() # gets users review

    label, conf = predict_sentiment(model, review) #  predict sentiment and return confidence
    summary = summarise_text(review, num_sentences=2) # generate 2 line summary

    label_readable = "Positive" if label == "pos" else "Negative" # convert pos and neg into full words for user

    # outputting results to the user
    print("\n--- Result ---") 
    print(f"Sentiment: {label_readable} (confidence: {conf:.2f})")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__": # prevents accidental execution
    main()