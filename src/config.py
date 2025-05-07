from pathlib import Path

# Base model to use - we'll use a small but capable model
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Directory settings
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "models" / "electrical-engineering-assistant"

# Dataset configuration
DATASET_CONFIG = {
    'train_file': str(DATA_DIR / "train.json"),
    'validation_file': str(DATA_DIR / "validation.json"),
    'test_file': str(DATA_DIR / "test.json")
}

# Training configuration with reduced memory usage
TRAINING_CONFIG = {
    'num_train_epochs': 3,
    'per_device_train_batch_size': 1,
    'per_device_eval_batch_size': 1,
    'warmup_steps': 50,
    'learning_rate': 1e-5,
    'gradient_accumulation_steps': 8,
    'max_length': 256,
    'logging_steps': 5,
    'evaluation_strategy': "steps",
    'eval_steps': 25,
    'save_strategy': "steps",
    'save_steps': 25,
}