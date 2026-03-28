# src/core/ml/config.py
import yaml
from pathlib import Path


REQUIRED_KEYS = {
    "data":     ["data_dir", "batch_size"],
    "model":    ["name"],
    "training": ["epochs", "lr", "lambda_resist"],
    "mlflow":   ["experiment_name", "run_name"],
}


def load_config(path):
    """Load and validate a YAML config file, returning a config dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    _validate(config, path)
    return config


def _validate(config, path):
    """Raise ValueError if any required top-level section or key is missing."""
    for section, keys in REQUIRED_KEYS.items():
        if section not in config:
            raise ValueError(f"Config '{path}' missing section: '{section}'")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Config '{path}' missing key: '{section}.{key}'")


if __name__ == "__main__":
    config = load_config("configs/exp001_baseline.yaml")

    print("Config loaded successfully:")
    for section, values in config.items():
        print(f"\n  [{section}]")
        for k, v in values.items():
            print(f"    {k}: {v}")