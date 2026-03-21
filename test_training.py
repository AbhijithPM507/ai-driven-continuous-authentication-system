from app import create_app
import numpy as np

app, socketio = create_app()

with app.app_context():
    from models.behavioral_models import (
        KEYSTROKE_DIM, MOUSE_DIM, COMBINED_DIM,
        EnsembleBehavioralClassifier
    )
    from utils.feature_extractor import BehavioralFeatureExtractor
    
    print(f"Dims: ks={KEYSTROKE_DIM} ms={MOUSE_DIM} combined={COMBINED_DIM}")
    
    # Create dummy genuine features using extractor
    extractor = BehavioralFeatureExtractor()
    _dummy_ks = [{'dwell_time': 80, 'flight_time': 150, 'key': 'a', 'timestamp': i} for i in range(100)]
    _dummy_ms = [{'x': i*10, 'y': i*5, 'timestamp': i, 'event_type': 'move', 'speed': 100} for i in range(100)]
    
    genuine_features = []
    for i in range(100):
        ks = extractor.extract_keystroke_features([_dummy_ks[i]])
        ms = extractor.extract_mouse_features([_dummy_ms[i]])
        combined = extractor.get_combined_features(ks, ms)
        genuine_features.append(combined)
    
    print(f"Created {len(genuine_features)} genuine feature samples")
    
    # Try training
    classifier = EnsembleBehavioralClassifier(1, app.config['MODELS_BASE_PATH'])  # user_id=1
    try:
        training_results = classifier.train_initial_models(genuine_features)
        print("TRAINING SUCCESS — dimensions are correct")
        print(f"Training results: {training_results}")
        
        # Try prediction
        pred = classifier.predict_ensemble([genuine_features[0]])
        print(f"PREDICTION SUCCESS — result: {pred}")
        print("SYSTEM READY — safe to calibrate")
    except Exception as e:
        print(f"TRAINING FAILED — {e}")
        print("DO NOT CALIBRATE YET — fix this error first")