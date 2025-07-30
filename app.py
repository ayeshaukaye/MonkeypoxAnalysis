
import streamlit as st
import joblib
import eli5
from eli5.formatters.html import format_as_html
import pandas as pd
import matplotlib.pyplot as plt
from eli5.sklearn import explain_prediction
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from clean_util import custom_cleaner
from wordcloud import WordCloud

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

le = joblib.load("label_encoder.pkl")

lr_pipe = joblib.load("lr_pipeline.pkl")
mb_pipe = joblib.load("mb_pipeline.pkl")
cb_pipe = joblib.load("cb_pipeline.pkl")
svc_pipe = joblib.load("svc_pipeline.pkl")

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

st.title("📊 Monkeypox Post Classifier")

acc_lr, acc_mb, acc_cb, acc_svc = joblib.load("accuracy_scores.pkl")
ad_lr, ad_mb, ad_cb, ad_svc = joblib.load("adjusted_accuracy_scores.pkl")

acc_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Multinomial Naive Bayes', 'Complement Naive Bayes', 'Linear SVC'],
    'Accuracy': [acc_lr, acc_mb, acc_cb, acc_svc],
    'Adjusted Accuracy': [ad_lr, ad_mb, ad_cb, ad_svc]
})

X_test_df = pd.DataFrame({'text': X_test, 'label': y_test})

# Split by class
no_stress_posts = X_test_df[X_test_df['label'] == 0]
stress_posts = X_test_df[X_test_df['label'] == 1]

sampled_no_stress = no_stress_posts.sample(n=6, random_state=23)
sampled_stress = stress_posts.sample(n=4, random_state=23)

# Combine and shuffle
final_posts = pd.concat([sampled_no_stress, sampled_stress]).sample(frac=1, random_state=42).reset_index(drop=True)

page = st.sidebar.radio("Choose a page", ["Detect Stress", "Understand the Model"])

if page == "Detect Stress":
    # expander + model chooser code
    st.markdown("Choose a post below. Models will predict whether **stress** is detected.")

    st.subheader('Dataset Used')
    st.markdown("The data used is a scraped collection of **Instagram post captions** related to monkeypox. The dataset was labelled to indicate whether the posts display signs of stress or anxiety. All models have been trained on this data.")
    st.markdown("All below predictions are based on the ***SVC model***, since it has the highest accuracy.")

    st.subheader("ELI5 Usage")
    with st.expander("ℹ️ How to Read ELI5 Explanations", expanded=False):
        st.markdown("""
    **An ELI5 explanation shows *why* the model predicted a post as _Stress_ or _No Stress_. Here’s how to read it:**
                    
    ---
    **1.  Words are highlighted in color**  
    - Highlighted words are the ones the model found *important* in the text.
    - The color shows whether each word *increases* or *decreases* the chance of “Stress”.

    **2. Look at the colors**  
    - :red[Red  words] push the score **away from the displayed prediction**.
    - :green[Green words] push the score **toward the displayed prediction**.
    - A stronger color indicates a stronger influence.

    **3. Word weights**  
    - All words have a score - it shows how strongly that word affected the result.
    - A bigger score shows a bigger impact.

    ---

    **If a word you expect isn’t highlighted, it may not have much impact in this model.**
    """)

    st.subheader('Choose a post below:')

    # Display
    for idx, row in final_posts.iterrows():
        icon = "😰" if row["label"] == 1 else "😊"

        with st.expander(f"{icon} Post {idx+1}", expanded=True):
            st.write(f"{idx+1}. [{row['label']}] {row['text']}")

            if st.button(f"Predict Stress for this Post", key=idx, type='primary'):
                pred = svc_pipe.predict([row['text']])
                st.write("**Predicted Value:**", le.inverse_transform(pred))
                st.write("**Original Label:**", le.inverse_transform([row['label']]))

                st.markdown("### ELI5 Explanation")
                # ELI5 explanation
                explanation = eli5.explain_prediction(
                    svc_pipe.named_steps['clf'],  
                    doc=row['text'],
                    vec=svc_pipe.named_steps['tfidf'],
                    top=10
                )
                html = eli5.format_as_html(explanation)
                # Display explanation as HTML
                st.components.v1.html(html, height=400, scrolling=True)

elif page == "Understand the Model":
    
    # st.subheader("idk smthnnn")
    st.markdown(""" The goal of this web app is to demonstrate how a simple NLP classifer works in a non technical manner. I will attempt to
                explain how these models were built, what happens in the background and the limitations of using linear models for NLP tasks.
    """)

    st.subheader("The Dataset")
    st.markdown("""The data used is a scraped collection of **Instagram post captions** related to monkeypox. 
                The dataset was labelled to indicate whether the posts display signs of stress or anxiety. All models have been trained on this data.""")

    st.markdown("""Here is a snippet of what the data looks like after basic encoding:""")
    
    st.write(X_test_df.sample(5, random_state=1))

    st.markdown("""**Dataset Limitation**: As is clear from above, some of the data includes posts that are irrelevant to the topic but are included in the dataset, mostly
                because of the use of the #monkeypox hashtag to reach a wider audience. The classifier must understand that these posts do 
                not talk about monkeypox and so are not indicative of any stress related to it. """)

    st.markdown("""Label is a binary column, where `0` indicates `No Stress Detected` and `1` indicates `Stress Detected`.""")
                
    st.subheader("How do models understand text?")
    st.markdown("""
Basic linear models (like the ones used here) **cannot understand raw text directly**.  
Text must first be **converted into numbers** so a model can process it mathematically.


### How is text converted?

We use **text vectorizers** to transform text into **numerical representations**.

The vectorizer used here is **TF-IDF** (Term Frequency–Inverse Document Frequency).  
TF-IDF assumes that the most useful information about a document comes from words that:

- Appear **frequently within that document**, but  
- Appear **less frequently across other documents**.

For example, an article about *mental health* might contain the word *stress* many times — but that word might not appear as often in other articles.  
TF-IDF captures this by weighing terms accordingly.  
The log factor in TF-IDF penalizes words that appear **too frequently as well as too rarely**, effectively highlighting the words that matter the most.


### Our text is numeric — now what?

Once the text is converted, it becomes **feature data** that can be used to **train models**.

**The models used here:**

- **Logistic Regression:** A simple, popular model for binary classification, well suited for text tasks.
- **Naive Bayes:** A fast, baseline model that assumes each word is independent of the others. (This is a simplification — in reality, words do relate to each other, but Naive Bayes often performs well anyway.)
- **Linear SVC (Support Vector Classifier):** Very effective for **high-dimensional** data like text, as it finds the optimal *hyperplane* that best separates the classes.


### About class imbalance

The dataset used here has **more “No Stress” posts** than “Stress” posts, creating a **class imbalance**.  
Because of this, I've also showed **adjusted accuracy** (balanced accuracy) to account for the imbalance and give a fairer view of how each model performs.


### Results

Below you’ll see each model’s performance on the test set, along with an explanation of how it made its predictions.
    """)

    st.subheader('Test Set Accuracies')
    st.table(acc_df)

    st.subheader('Model Comparison - Accuracy')

    fig, ax = plt.subplots()
    ax.bar(acc_df['Model'], acc_df['Adjusted Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.set_title('Model Accuracy Comparison')
    plt.xticks(rotation=15, ha='right')

    st.pyplot(fig)

    st.subheader('Model Comparison - F1 Score')

    f1_lr, f1_mb, f1_cb, f1_svc = joblib.load("f1_scores.pkl")

    labels = sorted(set(y_test))
    x = range(len(labels))
    width = 0.2 

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - 1.5 * width for i in x], f1_lr, width=width, label='Logistic Regression', color='skyblue')
    ax.bar([i - 0.5 * width for i in x], f1_mb, width=width, label='Multinomial NB', color='salmon')
    ax.bar([i + 0.5 * width for i in x], f1_cb, width=width, label='Complement NB', color='violet')
    ax.bar([i + 1.5 * width for i in x], f1_svc, width=width, label='Linear SVC', color='lightgreen')

    #value labels
    for idx, scores in zip([-1.5, -0.5, 0.5, 1.5], [f1_lr, f1_mb, f1_cb, f1_svc]):
        for i, score in zip(x, scores):
            ax.text(i + idx * width, score + 0.01, f"{score:.2f}", ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score per Class — Model Comparison')
    ax.legend(loc=4) #lower right
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    st.pyplot(fig)

    st.markdown("""
The effects of the class imbalance are clear in this chart, as the models are not able to learn as much about stress posts due to less data rows for that category.
    """)

# Combine all text
text = " ".join(stress_posts["text"])

# remove irrelevant
text = text.replace("Monkeypox", "")
text = text.replace("monkeypox", "")
text = text.replace("mpox", "")
text = text.replace("Mpox", "")
text = text.replace("virus", "")

# Create word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=150
).generate(text)
# wordcloud.to_file("wordcloud3.png")

st.subheader("Word Cloud for Stress Posts")
st.image("wordcloud3.png")

with st.expander("View Explanation"):
    st.markdown("The wordcloud displays the most commonly found words in stress posts.")
    st.markdown("""
                These include words like **`outbreak`, `spread`, `case`, `health emergency`** and **`international concern`** which are clear indicators of an emotionally charged post, 
                that focuses on the global spread of the disease and reflects feelings of stress or anxiety. 

                Do note: since this is a basic linear model, it simply looks at each word seperately, not as parts of a whole.
                Hence the model's limitations lie in the fact that it cannot learn from context and may miss certain nuances of the text.
                """)

st.markdown("""
### Limitations

- The current **TF-IDF vectorizer** does not capture relationships between words or their order.  
  This means important context and nuance is lost.

- As a result, words that don’t actually signal stress can still affect predictions.  
  For example, in post 5, the word *“written”* had a positive weight (+0.18) toward a **“Stress detected”** result, even though it’s irrelevant.  
  This highlights how simple frequency-based weighting can produce misleading signals.

---

### Future improvements

- Use a lightweight **transformer-based model** like **DistilBERT**, which captures word meaning **in context**, allowing the same word to have different meanings depending on surrounding words.

- Explore **semantic vectorizers** like **Word2Vec** or **Sentence-BERT**, which learn **dense embeddings** that better reflect the relationships between words and phrases.

""")
