"""Command-line demo (works without Streamlit). Run from project root: python demo/run_demo.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import joblib, pandas as pd
from src.config import MODEL_PATH, FEATURES

if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Run: python train.py")
pipe = joblib.load(MODEL_PATH)["pipeline"]
samples = {
    "Strong student": [8, 95, 85, 90, 7.5],
    "Average student": [4, 75, 60, 70, 7],
    "At-risk student": [1, 50, 35, 30, 5],
}
print(f"{'Student':16} {'Study':>5} {'Attend':>6} {'Prev':>5} {'Assign':>6} {'Sleep':>5}  Prediction  P(Pass)")
for name, v in samples.items():
    X = pd.DataFrame([v], columns=FEATURES)
    p = pipe.predict(X)[0]; pr = pipe.predict_proba(X)[0][1]
    print(f"{name:16} {v[0]:>5} {v[1]:>6} {v[2]:>5} {v[3]:>6} {v[4]:>5}  {'PASS' if p else 'FAIL':10}  {pr:.0%}")
