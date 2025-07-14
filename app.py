
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

page = st.sidebar.radio("Choose a page", ["Understand the Model", "Detect Stress"])

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
    - Highlighted words are the ones the model found *important* in your text.
    - The color shows whether each word *increases* or *decreases* the chance of “Stress”.

    **2. Look at the colors**  
    - 🟥 **Red  words** push the score **away from the displayed prediction**.
    - 🟩 **Green words** push the score **toward the displayed prediction**.
    - A stronger color indicates a stronger influence.

    **3. Word weights**  
    - All words have a score - it shows how strongly that word affected the result.
    - A bigger score shows a bigger impact.

    ---

    🛑 If a word you expect isn’t highlighted, it may not have much impact in this model.
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
    
    st.subheader("idk smthnnn")
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
Basic Linear Models (like the ones used here) cannot understand text directly. Text must be converted to numbers for the model to be able to process it.
                
##### How's that done?



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