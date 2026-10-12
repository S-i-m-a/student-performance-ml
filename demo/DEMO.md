# Demo Guide (2–3 minutes)

> The video itself must be recorded by YOU. This file is the script and steps.

## Before recording
```
pip install -r requirements.txt
python train.py
python evaluate.py
```
Open 3 windows: terminal, browser (after `streamlit run app.py`), and `figures/` folder.

## Recording tools
Windows: Win+G (Xbox Game Bar) or OBS. macOS: Shift+Cmd+5. Or Zoom/Meet "record" with screen share.

## Script
**0:00–0:20 Intro** – "This is Student Performance Prediction using a Decision Tree. It predicts Pass or Fail from study hours, attendance, previous score, assignment completion and sleep hours. The dataset is synthetic, not real student data."

**0:20–1:00 Evaluation** – Run `python evaluate.py`. Point at: DummyClassifier accuracy 0.500 vs Decision Tree 0.805; precision 0.785, recall 0.840, F1 0.812; confusion matrix [[77,23],[16,84]]. "The tree beats the baseline clearly."

**1:00–1:20 Figures** – Show `figures/decision_tree.png` and `feature_importance.png`.

**1:20–2:20 App** – Run `streamlit run app.py`. Enter these and click Predict:
| Case | Study | Attend | Prev | Assign | Sleep | Expected (from actual run) |
|---|---|---|---|---|---|---|
| Strong | 8 | 95 | 85 | 90 | 7.5 | PASS (~99%) |
| Average | 4 | 75 | 60 | 70 | 7 | PASS (~60%) |
| At-risk | 1 | 50 | 35 | 30 | 5 | FAIL (~1% pass) |

**2:20–2:40 Close** – "Predictions are educational estimates, not guarantees. Limitations: synthetic data and a small dataset. Future work: real data and other models."

## Backup if Streamlit fails
`python demo/run_demo.py` prints the same three predictions in the terminal.

## Saving the video
Videos are large: do NOT commit them to Git. Upload to YouTube (Unlisted) or Google Drive (anyone with link), then paste the link in README under "Demo video". Submit that link through the college form.
