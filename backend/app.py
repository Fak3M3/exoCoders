from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from tsfresh import extract_features
from src.predict import ModeloPredictdor
from src.utils import setup_logging, validate_features, format_prediction_response, log_prediction, health_check_model


app = Flask(__name__)
CORS(app)
loggin = setup_logging()
predictor = ModeloPredictdor(model_path="model/random_forest_model.pkl")

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
    
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file provided", "success": False}), 400
        df = pd.read_csv(file)
        feuture = extract_features(df, )
        validacion = validate_features(feuture, expected_count=150)

        if not validacion["valid"]:
            return jsonify({"error": validacion["error"], "success": False}), 400
        result = predictor.predict(feuture)
        response = format_prediction_response(result)

        if response["success"]:
            log_prediction(features=feuture, prediction=response["prediction"], user_id="anonymous")    

        return jsonify(response), 200 if response["success"] else 500

    except Exception as e:
        loggin.error(f"Error al cargar el archivo: {str(e)}")
        return jsonify({"error": "Error por parte del servidor"}),500
   
@app.route('/health')
def health():
    """Endpoint para verificar salud de la API y modelo"""
    try:
        model_status = health_check_model("model/random_forest_model.pkl")
        predictor_status = predictor.model is not None if predictor else False
        
        return jsonify({
            "api_status": "healthy",
            "model_loaded": predictor_status,
            "model_details": model_status,
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