import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class ModeloPredictor:
    def __init__(self, model_path: str):
        """Inicializar el predictor cargando el modelo"""
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Cargar el modelo Random Forest"""
        try:
            self.model = joblib.load(model_path)
            print(f"Modelo cargado desde: {model_path}")
        except FileNotFoundError:
            print(f"Error: No se encontró el modelo en {model_path}")
            self.model = None
    
    def preprocess_features(self, features: List[float]) -> np.ndarray:
        """Preprocesar las features antes de la predicción"""
        # Convertir a numpy array
        features_array = np.array(features).reshape(1, -1)
        
        # Aquí puedes agregar normalización, escalado, etc.
        # features_array = self.scaler.transform(features_array)
        
        return features_array
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """Hacer predicción con el modelo"""
        if self.model is None:
            return {"error": "Modelo no cargado", "success": False}
        
        try:
            # Preprocesar features
            processed_features = self.preprocess_features(features)
            
            # Hacer predicción
            prediction = self.model.predict(processed_features)
            probability = self.model.predict_proba(processed_features)
            
            return {
                "prediction": prediction[0],
                "probability": probability[0].tolist(),
                "success": True
            }
        
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtener importancia de las features"""
        if self.model is None:
            return {}
        
        try:
            importance = self.model.feature_importances_
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
        except:
            return {}