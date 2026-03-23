import joblib
import pandas as pd
from pathlib import Path
from .config import settings

_model = None
_feature_names = None

def load_model():
    """Load the model and its feature names."""
    global _model, _feature_names
    if _model is None:
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        _model = joblib.load(model_path)
        if hasattr(_model, 'feature_names_in_'):
            _feature_names = _model.feature_names_in_.tolist()
        else:
            raise ValueError("Model does not have 'feature_names_in_' attribute.")
    return _model, _feature_names
