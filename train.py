import joblib
from sklearn.model_selection import train_test_split
from src.config import *
from src.data import load
from src.model import build_pipeline

def main():
    df = load()
    print("Shape:", df.shape); print("Missing:\n", df.isna().sum())
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pipe = build_pipeline(seed=SEED).fit(Xtr, ytr)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "Xte": Xte, "yte": yte}, MODEL_PATH)
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    main()
