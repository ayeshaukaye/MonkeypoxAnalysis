
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from clean_util import custom_cleaner

# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# === Load Data ===
data = pd.read_excel(r"MonkeyPox.xlsx", sheet_name='English')

cleaner = FunctionTransformer(custom_cleaner)

le = LabelEncoder()
data["target"] = le.fit_transform(data["Stress or Anxiety"])
# Save encoder
joblib.dump(le, "label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(data["Post description"], data["target"], test_size=0.2, random_state=42)

joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

# === Pipelines ===
def make_pipeline(clf, tfidf_vectorizer=None):
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words='english', lowercase=True, ngram_range=(1, 2)
        )
    return Pipeline([
        ('cleaner', cleaner),
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True, ngram_range=(1, 2))),
        ('clf', clf)
    ])

pipelines = {
    "lr_pipeline.pkl": make_pipeline(
        LogisticRegression(max_iter=1000, C=10, class_weight='balanced'),
        TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True, ngram_range=(1, 1))
        ),
    "mb_pipeline.pkl": make_pipeline(MultinomialNB()),
    "cb_pipeline.pkl": make_pipeline(ComplementNB()),
    "svc_pipeline.pkl": make_pipeline(LinearSVC(dual=True, max_iter=5000, class_weight='balanced'))
}

# === Train + Save Models ===
for filename, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, filename)
    print(f"Saved: {filename}")

'''
y_pred_lr = lr_pipe.predict(X_test)
y_pred_mb = mb_pipe.predict(X_test)
y_pred_cb = cb_pipe.predict(X_test)
y_pred_svc = svc_pipe.predict(X_test)


acc_lr = accuracy_score(y_test, y_pred_lr)
acc_mb = accuracy_score(y_test, y_pred_mb)
acc_cb = accuracy_score(y_test, y_pred_cb)
acc_svc = accuracy_score(y_test, y_pred_svc)
'''