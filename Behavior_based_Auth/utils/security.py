import hashlib
import hmac
import os
import json
import logging

logger = logging.getLogger(__name__)

MODEL_HASH_FILE = 'model_hashes.json'

def compute_file_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def sign_model_file(filepath: str, key: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return hmac.new(key.encode(), sha256.digest(), hashlib.sha256).hexdigest()

def verify_model_integrity(filepath: str, expected_hash: str) -> bool:
    actual = compute_file_hash(filepath)
    return hmac.compare_digest(actual, expected_hash)

def save_model_hashes(models_dir: str, key: str, hash_path: str = None):
    if hash_path is None:
        hash_path = os.path.join(models_dir, MODEL_HASH_FILE)
    hashes = {}
    for fname in os.listdir(models_dir):
        fpath = os.path.join(models_dir, fname)
        if os.path.isfile(fpath) and not fname.endswith('.json'):
            hashes[fname] = sign_model_file(fpath, key)
    with open(hash_path, 'w') as f:
        json.dump(hashes, f, indent=2)
    logger.info(f"Saved {len(hashes)} model hashes to {hash_path}")

def load_and_verify_models(models_dir: str, key: str, hash_path: str = None) -> bool:
    if hash_path is None:
        hash_path = os.path.join(models_dir, MODEL_HASH_FILE)
    if not os.path.exists(hash_path):
        logger.warning("No model hash file found — skipping integrity check")
        return True
    with open(hash_path) as f:
        stored_hashes = json.load(f)
    for fname, expected in stored_hashes.items():
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            logger.error(f"Model file missing: {fpath}")
            return False
        actual = sign_model_file(fpath, key)
        if not hmac.compare_digest(actual, expected):
            logger.error(f"Model integrity check FAILED for: {fname}")
            return False
    logger.info("All model integrity checks passed")
    return True
