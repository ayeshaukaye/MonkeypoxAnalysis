
import streamlit as st
import joblib
import eli5
from eli5.sklearn import explain_prediction
from sklearn.metrics import accuracy_score

from clean_util import custom_cleaner

le = joblib.load("label_encoder.pkl")

lr_pipe = joblib.load("lr_pipeline.pkl")
mb_pipe = joblib.load("mb_pipeline.pkl")
cb_pipe = joblib.load("cb_pipeline.pkl")
svc_pipe = joblib.load("svc_pipeline.pkl")

tfidf = lr_pipe.named_steps['tfidf']
clf  = svc_pipe.named_steps['clf']

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

st.title("📊 Monkeypox Post Classifier")
st.markdown("Enter a tweet/post below. Models will predict whether **stress** is detected.")

user_input = st.text_area("Post Content")

if st.button("Predict Stress"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        pred_lr = le.inverse_transform(lr_pipe.predict([user_input]))[0]
        pred_mb = le.inverse_transform(mb_pipe.predict([user_input]))[0]
        pred_cb = le.inverse_transform(cb_pipe.predict([user_input]))[0]
        pred_svc =le.inverse_transform(svc_pipe.predict([user_input]))[0]

        st.markdown("### 🔍 Predictions")
        st.write(f"**Logistic Regression:** `{pred_lr}`")
        proba = lr_pipe.predict_proba([user_input])[0]
        st.write(f"Confidence (Stress): {proba[1]:.2%}")
        expl = eli5.format_as_html(
            eli5.explain_prediction(clf, user_input, vec=tfidf,feature_names=tfidf.get_feature_names_out())
            )


        st.write(f"**Multinomial NB:** `{pred_mb}`")
        st.write(f"**Complement NB:** `{pred_cb}`")
        st.write(f"**Linear SVC:** `{pred_svc}`")

        st.markdown("### Top Influential Words")
        st.components.v1.html(expl, height=400, scrolling=True)



acc_lr = accuracy_score(y_test, lr_pipe.predict(X_test))
acc_mb = accuracy_score(y_test, mb_pipe.predict(X_test))
acc_cb = accuracy_score(y_test, cb_pipe.predict(X_test))
acc_svc = accuracy_score(y_test, svc_pipe.predict(X_test))

st.subheader("📊 Model Accuracies on Test Set")
st.write(f"**Logistic Regression:** `{acc_lr:.4f}`")
st.write(f"**Multinomial NB:** `{acc_mb:.4f}`")
st.write(f"**Complement NB:** `{acc_cb:.4f}`")
st.write(f"**Linear SVC:** `{acc_svc:.4f}`")