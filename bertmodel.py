from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("loading data")
df = pd.read_excel(r"MonkeyPox.xlsx", sheet_name='English')
df = df[['Post description', 'Stress or Anxiety']]
le = LabelEncoder()
df["labels"] = le.fit_transform(df["Stress or Anxiety"])
df = df.rename(columns={
    'Post description': 'text'
    })

print(df.head())

data = Dataset.from_pandas(df)

# raw text -> token IDs for BERT
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=128)

print("tokenzing data")
tokenized_data = data.map(tokenize_function, batched=True)
print("splitting data")
tokenized_data = tokenized_data.train_test_split(test_size=0.2)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
)
print("model training start")
trainer.train()

print("model training end")
model.save_pretrained("./distilbertmodel")
tokenizer.save_pretrained("./distilbertmodel")

from transformers import pipeline

classifier = pipeline("text-classification", model="./distilbertmodel")

result = classifier("I am so stressed about monkeypox!!")
print(result)
