import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def custom_cleaner(X):
    cleaned = []
    for txt in X:
        txt = re.sub(r"http\S+|www\S+", " ", txt)
        txt = re.sub(r"[^A-Za-z0-9\s]", " ", txt)
        tokens = word_tokenize(txt)
        tokens = [wnl.lemmatize(w) for w in tokens]
        cleaned.append(" ".join(tokens))
    return cleaned
