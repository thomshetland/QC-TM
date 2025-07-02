import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['qc-tm']

def get_training_config(config_path="config.yaml"):
    """Get training parameters from config."""
    config = load_config(config_path)
    return config.get('training', {})

def get_preprocessing_config(config_path="config.yaml"):
    """Get preprocessing parameters from config."""
    config = load_config(config_path)
    return config.get('preprocessing', {})