import unittest, pandas as pd
from src.data import generate
from src.model import build_pipeline
from src.config import FEATURES, TARGET

class T(unittest.TestCase):
    def test_no_leakage_and_columns(self):
        d = generate(200)
        self.assertEqual(set(FEATURES + [TARGET]), set(d.columns))
    def test_imputation_handles_nan(self):
        d = generate(300); self.assertTrue(d[FEATURES].isna().any().any())
        p = build_pipeline().fit(d[FEATURES], d[TARGET]); self.assertEqual(len(p.predict(d[FEATURES])), 300)
    def test_prediction_valid(self):
        d = generate(300); p = build_pipeline().fit(d[FEATURES], d[TARGET])
        x = pd.DataFrame([[8, 95, 85, 90, 7]], columns=FEATURES)
        self.assertIn(p.predict(x)[0], (0, 1)); self.assertAlmostEqual(p.predict_proba(x)[0].sum(), 1.0)
    def test_reproducible(self):
        self.assertTrue(generate(50).equals(generate(50)))
if __name__ == "__main__":
    unittest.main()
