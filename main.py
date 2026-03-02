import time
import os
import pandas as pd
from utils.feature_extractor import extract_features
from fusion.fusion_engine import FusionEngine
from runtime.state_manager import StateManager

DATA_FILE = "data/live_session.csv"

def main():

    fusion = FusionEngine()
    state_manager = StateManager(fusion.threshold)

    print("🚀 Continuous Authentication Started")

    while True:

        try:
            if not os.path.exists(DATA_FILE):
                time.sleep(2)
                continue

            df = pd.read_csv(DATA_FILE)

            if len(df) < 10:
                time.sleep(2)
                continue

            key_feat, mouse_feat, click_feat = extract_features(df)

            if key_feat is None or mouse_feat is None or click_feat is None:
                time.sleep(2)
                continue

            # 🔥 Build expected feature dictionary
            features = {
                "keystroke": key_feat,
                "mouse": mouse_feat,
                "click": click_feat
            }

            final_score = fusion.compute_score(features)

            state = state_manager.update(final_score)

            print(f"Score: {final_score:.4f} | State: {state}")

            if state == "INTRUDER":
                print("⚠ Intruder Detected")

        except Exception as e:
            print("Error:", e)

        time.sleep(5)


if __name__ == "__main__":
    main()