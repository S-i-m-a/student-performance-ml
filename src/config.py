from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "students.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
FIG_DIR = ROOT / "figures"
SEED = 42
FEATURES = ["study_hours", "attendance", "previous_score", "assignment_completion", "sleep_hours"]
TARGET = "result"  # 1 = Pass, 0 = Fail
