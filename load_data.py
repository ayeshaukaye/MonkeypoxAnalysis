import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_excel(r"MonkeyPox.xlsx", sheet_name='English')

le = LabelEncoder()
data["target"] = le.fit_transform(data["Stress or Anxiety"])
# save encoder
joblib.dump(le, "label_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(data["Translated Post Description"], data["target"], test_size=0.2, random_state=42)

joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(X_train, "X_train.pkl")
joblib.dump(y_train, "y_train.pkl")

