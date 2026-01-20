from nltk.corpus import movie_reviews # imports dataset
from sklearn.model_selection import train_test_split # a function used to shuffle and split dataset into training and testing while keeping labels aligned
from sklearn.feature_extraction.text import TfidfVectorizer # an object used to convert text into numerical features using TF-IDF weighting

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
        random_state=42, # makes results repeatable with set randomness seed
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