import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.tree import plot_tree
from src.config import *
from src.data import load

def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run: python train.py")
    b = joblib.load(MODEL_PATH); pipe, Xte, yte = b["pipeline"], b["Xte"], b["yte"]
    FIG_DIR.mkdir(exist_ok=True)
    df = load()
    base = DummyClassifier(strategy="most_frequent").fit(Xte, yte)  # baseline
    for name, m in [("DummyClassifier", base), ("DecisionTree", pipe)]:
        p = m.predict(Xte)
        print(f"{name}: acc={accuracy_score(yte,p):.3f} prec={precision_score(yte,p,zero_division=0):.3f} "
              f"rec={recall_score(yte,p,zero_division=0):.3f} f1={f1_score(yte,p,zero_division=0):.3f}")
    p = pipe.predict(Xte)
    print("\nConfusion matrix [[TN FP],[FN TP]]:\n", confusion_matrix(yte, p))
    print("\n", classification_report(yte, p, target_names=["Fail", "Pass"]))
    tree = pipe.named_steps["tree"]
    # figures
    plt.figure(); sns.countplot(x=df[TARGET].map({0:"Fail",1:"Pass"})); plt.title("Class distribution")
    plt.savefig(FIG_DIR/"class_distribution.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); sns.heatmap(confusion_matrix(yte,p), annot=True, fmt="d", cmap="Blues",
        xticklabels=["Fail","Pass"], yticklabels=["Fail","Pass"]); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion matrix"); plt.savefig(FIG_DIR/"confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); sns.barplot(x=tree.feature_importances_, y=FEATURES); plt.title("Feature importance")
    plt.savefig(FIG_DIR/"feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(18,8)); plot_tree(tree, feature_names=FEATURES, class_names=["Fail","Pass"], filled=True, fontsize=7)
    plt.savefig(FIG_DIR/"decision_tree.png", dpi=150, bbox_inches="tight"); plt.close()
    print("Figures saved to", FIG_DIR)

if __name__ == "__main__":
    main()
