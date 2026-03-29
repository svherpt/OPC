# src/core/ml/models/__init__.py
"""Import all models to ensure they register themselves before build_model is called."""
from src.core.ml.models import unet_film, unet_film_v2