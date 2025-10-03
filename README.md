# Monkeypox Analysis - Streamlit Application
A Streamlit application designed as a Monkeypox Sentiment Classifier.

Built using several ML pipelines to detect sentiment in social media posts, and deployed for non-technical audience use and prediction explainability.

## Built With
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-202228?style=for-the-badge&logo=nltk&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## Try it out!
https://monkeypoxanalysis.streamlit.app/

## 

### 1. The Dataset
 - A scraped collection of 30,000+ Instagram post captions related to monkeypox, containing several irrelevant/spammy posts as is the nature of social media data.
### 2. Text Preprocessing using NLTK
 - URLs and special characters were removed using `re` module
 - Each post's content was then tokenized and lemmatized
### 3. Model Pipelines
 - Called cleaner function created in step 2 on each row
 - Used `TfidfVectorizer` to vectorize the text data
 - Called classifier function
 - Created pipelines for each of the 4 models, trained on data and saved models as `.pkl` files

``` def make_pipeline(clf, tfidf_vectorizer=None):
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words='english', lowercase=True, ngram_range=(1, 2)
        )
    return Pipeline([
        ('cleaner', cleaner),
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True, ngram_range=(1, 2))),
        ('clf', clf)
    ])
```
### 4. Implement in streamlit & explain predictions
 - When user presses the `Predict Stress` button, the SVC model predicts stress for the particular post
 - `eli5` showcases the most deterministic words that pushed the result 


## Roadmap

- [ ]  Parse eli5 output more intuitively for non technical users
- [ ]  Explore semantic vectorizers
- [ ]  Explore a lightweight transformer-based model
- [ ]  Interactive Visualizations
- [ ]  Restructure text-heavy sections
- [ ]  UI Improvements
