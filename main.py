# main.py

import time
from fusion.fusion_engine import FusionEngine
from runtime.state_manager import StateManager
from utils.feature_extractor import extract_features


def main():

    fusion = FusionEngine()
    state_manager = StateManager(fusion.threshold)

    print("🚀 Continuous Authentication Started")

    while True:

        features = extract_features()

        if features is None:
            time.sleep(2)
            continue

        final_score = fusion.compute_score(features)

        state = state_manager.update(final_score)

        print(f"Score: {final_score:.4f} | State: {state}")

        if state == "INTRUDER":
            print("⚠ Intruder Detected")

        time.sleep(5)


if __name__ == "__main__":
    main()
