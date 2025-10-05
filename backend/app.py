from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from tsfresh import extract_features
from src.predict import ModeloPredictor
from src.utils import setup_logging, validate_features, format_prediction_response, log_prediction, health_check_model, apply_full_preprocessing_pipeline, get_model_metrics

app = Flask(__name__)
CORS(app)
loggin = setup_logging()

# Cargar modelo y componentes de preprocesamiento
try:
    # Verificar que el modelo existe antes de cargar
    model_path = "model/random_forest_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    
    predictor = ModeloPredictor(model_path=model_path)
    print(f"✅ Predictor inicializado correctamente")
    
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("🔧 Asegúrate de ejecutar el entrenamiento primero:")
    print("   cd ml-entrenamiento && python main.py")
    exit(1)
except Exception as e:
    print(f"❌ Error cargando componentes: {e}")
    exit(1)

@app.route('/')
def home():
    """Endpoint raíz para verificar que la API está funcionando"""
    return jsonify({
        "message": "ExoCoders ML API - Predicción con Random Forest",
        "status": "operacional",
        "version": "1.0.0",
        "endpoints": {
            "predict": {
                "url": "/predict",
                "method": "POST",
                "params": {"file": "CSV file"},
                "description": "Analiza archivo CSV y hace predicción con ML"
            },
            "health": {
                "url": "/health", 
                "method": "GET",
                "description": "Verifica estado de la API y modelo"
            }
        }
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de predicción con modelo real únicamente"""
    try:
        # 1. Verificar archivo
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided", "success": False}), 400
        
        # 2. Leer CSV
        df = pd.read_csv(file)
        print(f"📄 Archivo recibido: {df.shape}")
        
        # 3. Aplicar pipeline completo de preprocesamiento
        processed_features = apply_full_preprocessing_pipeline(df)
        
        # 4. Validar características
        expected_features = len(selected_features) if selected_features else 200
        validation = validate_features(processed_features, expected_features)
        
        if not validation["valid"]:
            return jsonify({
                "error": validation["error"], 
                "success": False,
                "validation_details": validation
            }), 400
        
        # 5. Hacer predicción con modelo real
        result = predictor.predict(processed_features)
        
        if not result["success"]:
            return jsonify(result), 500
        
        # 6. Formatear respuesta
        response = format_prediction_response(result)
        
        # 7. Log de la predicción
        if response["success"]:
            log_prediction(
                features=processed_features.tolist(), 
                prediction=response.get("predictions"), 
                user_id="anonymous"
            )
        
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error procesando archivo: {str(e)}")
        return jsonify({
            "error": f"Error del servidor: {str(e)}", 
            "success": False
        }), 500

@app.route('/health')
def health():
    """Endpoint de salud - solo modelo real"""
    try:
        # Verificar estado del modelo
        model_status = health_check_model("model/random_forest_model.pkl")
        
        # Verificar estado del predictor
        predictor_info = predictor.get_model_info() if predictor else {"error": "Predictor no cargado"}
        
        # Obtener métricas del modelo
        model_metrics = get_model_metrics()
        
        return jsonify({
            "api_status": "healthy",
            "model_status": model_status,
            "predictor_info": predictor_info,
            "model_metrics": model_metrics,
            "preprocessing_status": {
                "preprocessor_loaded": preprocessor is not None,
                "feature_engineer_loaded": feature_engineer is not None,
                "selected_features_count": len(selected_features) if selected_features else 0
            },
            "timestamp": pd.Timestamp.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "api_status": "unhealthy",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)