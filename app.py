import joblib, pandas as pd, streamlit as st
from src.config import MODEL_PATH, FEATURES

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Prediction")
st.caption("Decision Tree model trained on a SYNTHETIC dataset (not real student data).")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train.py` first, then restart the app."); st.stop()
pipe = joblib.load(MODEL_PATH)["pipeline"]

c1, c2 = st.columns(2)
study = c1.slider("Study hours per day", 0.0, 10.0, 4.0, 0.5)
att = c1.slider("Attendance (%)", 0.0, 100.0, 75.0)
prev = c2.slider("Previous exam score", 0.0, 100.0, 60.0)
asg = c2.slider("Assignment completion (%)", 0.0, 100.0, 70.0)
sleep = c2.slider("Sleep hours per day", 3.0, 10.0, 7.0, 0.5)

if st.button("Predict"):
    try:
        X = pd.DataFrame([[study, att, prev, asg, sleep]], columns=FEATURES)
        pred = pipe.predict(X)[0]; proba = pipe.predict_proba(X)[0]
        (st.success if pred == 1 else st.error)("Prediction: **PASS**" if pred == 1 else "Prediction: **FAIL**")
        st.write(f"Model confidence (leaf class proportion): Pass {proba[1]:.0%} | Fail {proba[0]:.0%}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.expander("How the model works"):
    st.write("A Decision Tree asks a series of yes/no questions about the inputs (e.g. 'previous score > 55?') "
             "and follows branches until it reaches a leaf giving Pass or Fail. Splits are chosen to reduce Gini impurity.")
st.info("Disclaimer: predictions are educational estimates, not guarantees about any student's future.")
