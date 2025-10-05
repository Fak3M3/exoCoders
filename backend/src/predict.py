import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
import os

class ModeloPredictor:
    def __init__(self, model_path: str):
        """Inicializar el predictor cargando el modelo LightGBM"""
        self.model = None
        self.model_path = model_path
        self.is_lightgbm = "lightgbm" in model_path.lower()
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Cargar el modelo"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modelo no encontrado en: {model_path}")
        
        try:
            self.model = joblib.load(model_path)
            model_type = "LightGBM" if self.is_lightgbm else type(self.model).__name__
            print(f"✅ Modelo {model_type} cargado exitosamente")
            
        except Exception as e:
            raise Exception(f"❌ Error cargando modelo: {str(e)}")
    
    def predict(self, data: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Hacer predicción con el modelo"""
        if self.model is None:
            return {"error": "Modelo no cargado", "success": False}
        
        try:
            processed_data = self._prepare_data(data)
            print(f"🔮 Haciendo predicción con {processed_data.shape}")
            
            # ✅ Predicción según tipo de modelo:
            if self.is_lightgbm:
                # LightGBM prediction
                y_prob = self.model.predict(processed_data, num_iteration=self.model.best_iteration)
                predictions = (y_prob >= 0.5).astype(int)
                probabilities = np.column_stack([1-y_prob, y_prob])  # [prob_class_0, prob_class_1]
            else:
                # Scikit-learn prediction
                predictions = self.model.predict(processed_data)
                probabilities = self.model.predict_proba(processed_data)
            
            result = {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "success": True,
                "input_shape": processed_data.shape,
                "exoplanets_detected": int(np.sum(predictions)),
                "total_samples": len(predictions),
                "model_type": "LightGBM" if self.is_lightgbm else type(self.model).__name__
            }
            
            print(f"✅ Predicción completada: {result['exoplanets_detected']}/{result['total_samples']} exoplanetas detectados")
            return result
            
        except Exception as e:
            return {
                "error": f"Error en predicción: {str(e)}", 
                "success": False
            }
    
    def _prepare_data(self, data: Union[List[float], np.ndarray]) -> np.ndarray:
        """Preparar datos para predicción"""
        if isinstance(data, list):
            # Lista de características para una muestra
            if all(isinstance(x, (int, float)) for x in data):
                return np.array(data).reshape(1, -1)
            # Lista de listas (múltiples muestras)
            else:
                return np.array(data)
        
        elif isinstance(data, np.ndarray):
            # Array numpy
            if data.ndim == 1:
                return data.reshape(1, -1)
            else:
                return data
        
        else:
            raise ValueError(f"Formato de datos no soportado: {type(data)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtener importancia de las características"""
        if self.model is None:
            return {"error": "Modelo no cargado"}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return {
                    f"feature_{i}": float(imp) 
                    for i, imp in enumerate(importance)
                }
            else:
                return {"error": "Modelo no soporta feature_importances_"}
                
        except Exception as e:
            return {"error": f"Error obteniendo importancia: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if self.model is None:
            return {"error": "Modelo no cargado"}
        
        info = {
            "model_type": "LightGBM" if self.is_lightgbm else type(self.model).__name__,
            "model_path": self.model_path,
            "model_loaded": True
        }
        
        # Agregar información específica del modelo
        if hasattr(self.model, 'n_features_in_'):
            info["n_features"] = self.model.n_features_in_
        
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators
        
        if hasattr(self.model, 'max_depth'):
            info["max_depth"] = self.model.max_depth
        
        if hasattr(self.model, 'classes_'):
            info["classes"] = self.model.classes_.tolist()
        
        return info