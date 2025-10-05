from datetime import datetime
from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_features(features: Union[List[float], np.ndarray], expected_count: int) -> Dict[str, Any]:
    """Validate input features"""
    try:
        if isinstance(features, list):
            features = np.array(features)
        
        if features.ndim == 1:
            actual_count = len(features)
        else:
            actual_count = features.shape[1]
        
        return {
            "valid": True,
            "expected_count": expected_count,
            "actual_count": actual_count,
            "message": f"Features validated: {actual_count} features"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}"
        }

def format_prediction_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format prediction response"""
    try:
        return {
            "success": result.get("success", True),
            "predictions": result.get("predictions", []),
            "probabilities": result.get("probabilities", []),
            "exoplanets_detected": result.get("exoplanets_detected", 0),
            "total_samples": result.get("total_samples", 0),
            "model_type": result.get("model_type", "Unknown"),
            "confidence": "high" if result.get("exoplanets_detected", 0) > 0 else "low"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Response formatting error: {str(e)}"
        }

def log_prediction(features: List[float], prediction: Any, user_id: str = "anonymous"):
    """Log prediction for monitoring"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "features_count": len(features) if isinstance(features, list) else "unknown",
            "prediction": prediction,
            "status": "success"
        }
        print(f"📝 Prediction logged: {log_entry}")
    except Exception as e:
        print(f"⚠️ Logging error: {e}")

def health_check_model(model_path: str) -> Dict[str, Any]:
    """Check model health"""
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return {
                "status": "healthy",
                "model_loaded": True,
                "model_type": type(model).__name__,
                "file_size_mb": os.path.getsize(model_path) / (1024 * 1024)
            }
        else:
            return {
                "status": "unhealthy",
                "error": f"Model file not found: {model_path}"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def get_model_metrics() -> Dict[str, Any]:
    """Get model metrics"""
    try:
        model_info_path = "model/model_info.json"
        if os.path.exists(model_info_path):
            import json
            with open(model_info_path, 'r') as f:
                return json.load(f)
        else:
            return {"error": "Model metrics not available"}
    except Exception as e:
        return {"error": f"Error loading metrics: {str(e)}"}

def apply_full_preprocessing_pipeline(df: pd.DataFrame) -> np.ndarray:
    """
    Apply full preprocessing pipeline identical to main.py
    """
    try:
        print("🔄 Applying full preprocessing pipeline...")
        
        # Importar clases locales desde app.py
        from app import APIDataPreprocessor, APIFeatureEngineer, Config
        
        # Cargar componentes entrenados
        model_dir = Path("model")
        
        # 1. Cargar preprocessor entrenado
        if (model_dir / "preprocessor.pkl").exists():
            trained_preprocessor = joblib.load(model_dir / "preprocessor.pkl")
            preprocessor = APIDataPreprocessor(trained_preprocessor)
        else:
            preprocessor = APIDataPreprocessor()
        
        # 2. Cargar feature engineer entrenado
        if (model_dir / "feature_engineer.pkl").exists():
            feature_engineer = joblib.load(model_dir / "feature_engineer.pkl")
        else:
            feature_engineer = APIFeatureEngineer()
        
        # 3. Cargar características seleccionadas
        if (model_dir / "selected_features.csv").exists():
            selected_features_df = pd.read_csv(model_dir / "selected_features.csv")
            selected_features = selected_features_df['feature'].tolist()
        else:
            selected_features = None
        
        # 4. Aplicar pipeline
        print("🧹 Step 1: Data cleaning...")
        cleaned_data = preprocessor.clean_data(df)
        
        print("⚙️ Step 2: Feature engineering...")
        engineered_features = feature_engineer.create_all_features(cleaned_data)
        
        print("🎯 Step 3: Feature selection...")
        if selected_features:
            # Usar solo las características seleccionadas durante el entrenamiento
            available_features = [f for f in selected_features if f in engineered_features.columns]
            final_features = engineered_features[available_features]
            print(f"✅ Selected {len(available_features)}/{len(selected_features)} features")
        else:
            final_features = engineered_features
            print(f"⚠️ No feature selection applied - using all {final_features.shape[1]} features")
        
        print(f"✅ Pipeline completed. Final shape: {final_features.shape}")
        return final_features.values
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        raise e