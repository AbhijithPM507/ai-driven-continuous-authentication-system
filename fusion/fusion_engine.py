# fusion/fusion_engine.py

import joblib
import numpy as np


class FusionEngine:

    def __init__(self):

        self.keystroke_model = joblib.load("models/keystroke_model.pkl")
        self.mouse_model = joblib.load("models/mouse_model.pkl")
        self.click_model = joblib.load("models/click_model.pkl")

        self.keystroke_scaler = joblib.load("models/keystroke_scaler.pkl")
        self.mouse_scaler = joblib.load("models/mouse_scaler.pkl")
        self.click_scaler = joblib.load("models/click_scaler.pkl")

        self.fusion_scaler = joblib.load("models/fusion_scaler.pkl")
        self.threshold = joblib.load("models/fusion_threshold.pkl")

        print("✅ Models loaded successfully")

    def compute_score(self, features):

        # 1️⃣ Keystroke
        X_k = self.keystroke_scaler.transform(features["keystroke"])
        score_k = -self.keystroke_model.decision_function(X_k)[0]

        # 2️⃣ Mouse
        X_m = self.mouse_scaler.transform(features["mouse"])
        score_m = -self.mouse_model.decision_function(X_m)[0]

        # 3️⃣ Click
        X_c = self.click_scaler.transform(features["click"])
        score_c = -self.click_model.decision_function(X_c)[0]

        # 4️⃣ Stack scores
        scores = np.array([[score_k, score_m, score_c]])

        # 5️⃣ Normalize
        normalized = self.fusion_scaler.transform(scores)

        normalized_k = normalized[0][0]
        normalized_m = normalized[0][1]
        normalized_c = normalized[0][2]

        # 6️⃣ Weighted fusion
        final_score = (
            0.5 * normalized_k +
            0.3 * normalized_m +
            0.2 * normalized_c
        )

        return final_score
