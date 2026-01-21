import nltk # used to split text into sentences
from sklearn.feature_extraction.text import TfidfVectorizer # imports vectorizer to convert text into TF-IDF weights

# used to summarise the text by selecting the 2 most informative sentences
def summarise_text(text: str, num_sentences: int = 2) -> str:

    sentences = nltk.sent_tokenize(text) # split text into sentences using NLTK

    if len(sentences) <= num_sentences: # if review is already short no need to summerise
        return text
    
    vectorizer = TfidfVectorizer(stop_words="english") # remove common, uninformative words
    tfidf_matrix = vectorizer.fit_transform(sentences) # convert eahc sentence into TF-IDF weights 

    sentence_scores = tfidf_matrix.sum(axis=1).A1 # score sentence based on total TF-IDF weight

    top_indices = sentence_scores.argsort()[-num_sentences:] # sort sentences into top scoring
    top_indices = sorted(top_indices) # preserve orginal sentence order

    summary = " ".join(sentences[i] for i in top_indices) # create final summary by joining most informative sentances

    return summary