# src/core/ml/registry.py
"""Model registry for looking up and instantiating models by name from config."""

MODEL_REGISTRY = {}

def register_model(name):
    """Decorator that registers a model class under the given name."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(config):
    """Instantiate and return the model specified by config['model']['name']."""
    name = config["model"]["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](config)