import joblib
from sklearn.metrics import f1_score

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

lr_pipe = joblib.load("lr_pipeline.pkl")
mb_pipe = joblib.load("mb_pipeline.pkl")
cb_pipe = joblib.load("cb_pipeline.pkl")
svc_pipe = joblib.load("svc_pipeline.pkl")

def f1():
    f1_lr = f1_score(y_test, lr_pipe.predict(X_test), average=None)
    f1_mb = f1_score(y_test, mb_pipe.predict(X_test), average=None)
    f1_cb = f1_score(y_test, cb_pipe.predict(X_test), average=None)
    f1_svc = f1_score(y_test, svc_pipe.predict(X_test), average=None)
    return f1_lr, f1_mb, f1_cb, f1_svc

f1_results = f1()
joblib.dump(f1_results, "f1_scores.pkl")
