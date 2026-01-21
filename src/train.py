from nltk.corpus import movie_reviews # imports dataset
from sklearn.model_selection import train_test_split # a function used to shuffle and split dataset into training and testing while keeping labels aligned
from sklearn.feature_extraction.text import TfidfVectorizer # an object used to convert text into numerical features using TF-IDF weighting
from sklearn.linear_model import LogisticRegression # classifier used to predict posotive or negative sentiment
from sklearn.pipeline import Pipeline # helper class used to build pipeline
from sklearn.metrics import accuracy_score, classification_report # two helper functions used to evaluate model

# loading dataset into lists with each text corresponding to a label
def load_dataset():
    texts = [] # list of review texts (strings)
    labels = [] # list of sentiment labels ("pos" or "neg")

    for fileid in movie_reviews.fileids(): # loops through every review
        texts.append(movie_reviews.raw(fileid)) # retrieves the review corresponding to the id
        labels.append(movie_reviews.categories(fileid)[0]) # retrieves the sentiment label corresponding to the id

    return texts, labels

# split data fairly into training data and test data to prevent misleading accuracy by evaluating on unseen reviews
def split_dataset(texts, labels): # parameters are the loaded dataset
    texts_train, texts_test, labels_train, labels_test = train_test_split( # new lists to store split data 
        texts,
        labels,
        test_size=0.2, # 20% test 80% training
        random_state=42, # ensures the results are reproducible with set randomness seed
        stratify=labels # preserves proportion between positive and negative
    )

    return texts_train, texts_test, labels_train, labels_test

# create a TF-IDF vectorizer to help convert text into numbers
def create_vectorizer():
    vectorizer = TfidfVectorizer(
        stop_words="english", # ignore common words that do not help sentiment classification
        max_features=20000 # limit vocabulary size to reduce noise and memory usage
    )
    return vectorizer

#learns feature weights through iteration to classify what words are positive or negative
def create_model():
    model = LogisticRegression(
        max_iter=1000 # prevent warnings by capping iterations
    )
    return model

# creates the pipeline to apply TF-IDF by classification in a fixed order
def create_pipeline(): # prevents data leakage by ensuring process applied consistently and only fitted on data given
    pipeline = Pipeline([
        ("tfidf", create_vectorizer()),
        ("classifier", create_model())
    ])
    return pipeline

# fits the training data and evaluates the model
def train_and_evaluate():
    texts, labels = load_dataset() # load dataset
    X_train, X_test, y_train, y_test = split_dataset(texts, labels) # split dataset

    pipeline = create_pipeline() # create pipeline
    pipeline.fit(X_train, y_train) # train the pipeline on the data

    predictions = pipeline.predict(X_test) # predicts positive or negative

    print("Accuracy:", accuracy_score(y_test, predictions)) # prints accuracy (0-1)
    print(classification_report(y_test, predictions)) # prints precision, recall and f1-score for each class

if __name__ == "__main__": # only runs code if file executed directly, prevents accidental training
    train_and_evaluate()