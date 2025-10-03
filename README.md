# Monkeypox Analysis - Streamlit Application
A Streamlit application designed as a Monkeypox Stress Classifier.

Built using several ML pipelines to detect stress in social media posts using NLP techniques, and deployed for non-technical audience use and prediction explainability.

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
 - The pre-processing function (from step 2) was applied to the data
 - The cleaned text was vectorized using `TfidfVectorizer` (configured to use bigrams, remove English stop words, and select the top 5,000 features)
 - Logistic Regression, Naive Bayes, and SVC models were trained and evaluated on the vectorized data
 - A Scikit-learn `Pipeline` was created for each model, combining the text-cleaning, TF-IDF vectorization, and classifier steps
 -  Trained models were saved as `.pkl` files for deployment

``` 
    return Pipeline([
        ('cleaner', cleaner),
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True, ngram_range=(1, 2))),
        ('clf', clf)
    ])
```
<img width="930" height="612" alt="image" src="https://github.com/user-attachments/assets/473119c9-941c-438f-a241-c9bd18c66e45" />

### 4. Implement in streamlit & explain predictions
 - Loaded the SVC model (chosen for its high performance after hyperparameter tuning and comparison with Naive Bayes/Logistic Regression) 
 - When user presses the `Predict Stress` button, the SVC model predicts stress for the particular post
 - `eli5` showcases the most deterministic words that contributed to the final result 
<img width="691" height="467" alt="image" src="https://github.com/user-attachments/assets/fa24f397-15bb-4b13-8e67-26ac673b9e55" />
<img width="767" height="726" alt="image" src="https://github.com/user-attachments/assets/a6d7ac11-ef12-4f30-ad6a-34ca90e15ae4" />

## Roadmap

- [ ]  Parse eli5 output more intuitively for non technical users
- [ ]  Explore semantic vectorizers
- [ ]  Explore a lightweight transformer-based model
- [ ]  Interactive Visualizations
- [ ]  Restructure text-heavy sections
- [ ]  UI Improvements
