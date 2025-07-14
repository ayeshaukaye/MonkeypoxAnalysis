import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score


X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

lr_pipe = joblib.load("lr_pipeline.pkl")
mb_pipe = joblib.load("mb_pipeline.pkl")
cb_pipe = joblib.load("cb_pipeline.pkl")
svc_pipe = joblib.load("svc_pipeline.pkl")

def acc(x=X_test, y=y_test):
    acc_lr = accuracy_score(y, lr_pipe.predict(x))
    acc_mb = accuracy_score(y, mb_pipe.predict(x))
    acc_cb = accuracy_score(y, cb_pipe.predict(x))
    acc_svc = accuracy_score(y, svc_pipe.predict(x))
    return acc_lr, acc_mb, acc_cb, acc_svc

def adj(x=X_test, y=y_test):
    ad_lr = balanced_accuracy_score(y, lr_pipe.predict(x))
    ad_mb = balanced_accuracy_score(y, mb_pipe.predict(x))
    ad_cb = balanced_accuracy_score(y, cb_pipe.predict(x))
    ad_svc = balanced_accuracy_score(y, svc_pipe.predict(x))
    return ad_lr, ad_mb, ad_cb, ad_svc

# Calculate once
acc_results = acc()
adj_results = adj()

print(acc_results)
print(adj_results)

# Save to disk
joblib.dump(acc_results, "accuracy_scores.pkl")
joblib.dump(adj_results, "adjusted_accuracy_scores.pkl")

