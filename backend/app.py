from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports para replicar el preprocesamiento del main.py
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks
from src.predict import ModeloPredictor
from src.utils import setup_logging, validate_features, format_prediction_response, log_prediction, health_check_model, get_model_metrics

# Agregar después de los imports y antes de cualquier función
# Crear aliases para compatibilidad con archivos pickle
DataPreprocessor = None  # Definir temporalmente
FeatureEngineer = None   # Definir temporalmente

# Las clases reales se definen más abajo, así que haremos el alias después

def apply_full_preprocessing_pipeline(df: pd.DataFrame) -> np.ndarray:
    """
    Apply full preprocessing pipeline identical to main.py
    """
    try:
        print("🔄 Applying full preprocessing pipeline...")
        
        # Cargar componentes entrenados
        model_dir = Path("model")
        
        # 1. Cargar preprocessor entrenado
        if (model_dir / "preprocessor.pkl").exists():
            trained_preprocessor = joblib.load(model_dir / "preprocessor.pkl")
            preprocessor_instance = APIDataPreprocessor(trained_preprocessor)
        else:
            preprocessor_instance = APIDataPreprocessor()
        
        # 2. Cargar feature engineer entrenado
        if (model_dir / "feature_engineer.pkl").exists():
            feature_engineer_instance = joblib.load(model_dir / "feature_engineer.pkl")
        else:
            feature_engineer_instance = APIFeatureEngineer()
        
        # 3. Cargar características seleccionadas
        if (model_dir / "selected_features.csv").exists():
            selected_features_df = pd.read_csv(model_dir / "selected_features.csv")
            selected_features_list = selected_features_df['feature'].tolist()
        else:
            selected_features_list = None
        
        # 4. Aplicar pipeline
        print("🧹 Step 1: Data cleaning...")
        cleaned_data = preprocessor_instance.clean_data(df)
        
        print("⚙️ Step 2: Feature engineering...")
        engineered_features = feature_engineer_instance.create_all_features(cleaned_data)
        
        print("🎯 Step 3: Feature selection...")
        if selected_features_list:
            # Usar solo las características seleccionadas durante el entrenamiento
            available_features = [f for f in selected_features_list if f in engineered_features.columns]
            final_features = engineered_features[available_features]
            print(f"✅ Selected {len(available_features)}/{len(selected_features_list)} features")
        else:
            final_features = engineered_features
            print(f"⚠️ No feature selection applied - using all {final_features.shape[1]} features")
        
        print(f"✅ Pipeline completed. Final shape: {final_features.shape}")
        return final_features.values
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        raise e

app = Flask(__name__)
CORS(app)
loggin = setup_logging()

try:
    predictor = ModeloPredictor(model_path="model/lightgbm_model.pkl")
    
    # Cargar componentes de preprocesamiento guardados desde el entrenamiento
    model_dir = Path("model")
    
    if (model_dir / "preprocessor.pkl").exists():
        preprocessor = joblib.load(model_dir / "preprocessor.pkl")
        print("✅ Preprocessor cargado")
    else:
        preprocessor = None
        print("⚠️ Preprocessor no encontrado")
    
    if (model_dir / "feature_engineer.pkl").exists():
        feature_engineer = joblib.load(model_dir / "feature_engineer.pkl")
        print("✅ Feature Engineer cargado")
    else:
        feature_engineer = None
        print("⚠️ Feature Engineer no encontrado")
    
    if (model_dir / "selected_features.csv").exists():
        selected_features_df = pd.read_csv(model_dir / "selected_features.csv")
        selected_features = selected_features_df['feature'].tolist()
        print(f"✅ {len(selected_features)} características seleccionadas cargadas")
    else:
        selected_features = None
        print("⚠️ Lista de características no encontrada")

except Exception as e:
    print(f"⚠️ Error cargando componentes: {e}")
    predictor = ModeloPredictor(model_path="model/lightgbm_model.pkl")
    preprocessor = None
    feature_engineer = None
    selected_features = None

# Configuración idéntica al main.py
class Config:
    OUTLIER_THRESHOLD = 3
    IMPUTATION_STRATEGY = 'median'
    ROLLING_WINDOWS = [5,10, 25, 50, 100]
    FFT_PERCENTILE = 85
    HISTOGRAM_BINS = 50

class APIDataPreprocessor:
    """
    REPLICA EXACTA del DataPreprocessor del main.py para la API
    """
    def __init__(self, trained_preprocessor=None):
        self.preprocessor = trained_preprocessor
        self.scaler = trained_preprocessor.scaler if trained_preprocessor else None
        self.imputer = trained_preprocessor.imputer if trained_preprocessor else None

    def clean_data(self, df):
        """MISMA LÓGICA que main.py DataPreprocessor.clean_data()"""
        print("🧹 Starting data cleaning...")
        
        df_clean = df.copy()
        
        # Si no hay columna LABEL, asumir que son solo características
        if 'LABEL' in df_clean.columns:
            X = df_clean.drop('LABEL', axis=1)
        else:
            X = df_clean
        
        print(f"📊 Input data shape: {X.shape}")
        
        # Handle missing values - IGUAL que main.py
        missing_count = X.isnull().sum().sum()
        print(f"🔍 Missing values found: {missing_count}")
        
        if missing_count > 0:
            if self.imputer is not None:
                # Usar imputer entrenado
                X_imputed = pd.DataFrame(
                    self.imputer.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                # Crear nuevo imputer si no hay entrenado
                imputer = SimpleImputer(strategy=Config.IMPUTATION_STRATEGY)
                X_imputed = pd.DataFrame(
                    imputer.fit_transform(X),
                    columns=X.columns,
                    index=X.index
                )
        else:
            X_imputed = X
        
        # Handle outliers - MISMA LÓGICA que main.py._handle_outliers()
        print("🎯 Handling outliers...")
        Q1 = X_imputed.quantile(0.25)
        Q3 = X_imputed.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - Config.OUTLIER_THRESHOLD * IQR
        upper_bound = Q3 + Config.OUTLIER_THRESHOLD * IQR
        
        X_clean = X_imputed.clip(lower=lower_bound, upper=upper_bound, axis=1)
        
        outlier_count = ((X_imputed < lower_bound) | (X_imputed > upper_bound)).sum().sum()
        print(f"✅ Outliers handled: {outlier_count}")
        
        print(f"✅ Data cleaning completed. Final shape: {X_clean.shape}")
        return X_clean

class APIFeatureEngineer:
    """REPLICA EXACTA del FeatureEngineer del main.py"""
    
    def create_statistical_features(self, X):
        print("📊 Creating statistical features...")
        features = pd.DataFrame(index=X.index)
        
        # Basic statistics - IGUAL que main.py
        features['mean_flux'] = X.mean(axis=1)
        features['median_flux'] = X.median(axis=1)
        features['std_flux'] = X.std(axis=1)
        features['var_flux'] = X.var(axis=1)
        features['min_flux'] = X.min(axis=1)
        features['max_flux'] = X.max(axis=1)
        features['range_flux'] = features['max_flux'] - features['min_flux']
        
        # ✅ AGREGAR quantiles que faltan:
        for q in [0.05, 0.25, 0.75, 0.95]:
            features[f'q{int(q*100)}_flux'] = X.quantile(q, axis=1)
        
        # Advanced statistics - IGUAL que main.py
        features['skewness'] = X.skew(axis=1)
        features['kurtosis'] = X.kurtosis(axis=1)
        
        # ✅ AGREGAR MAD que falta:
        features['mad'] = X.apply(lambda row: np.median(np.abs(row - row.median())), axis=1)
        
        # ✅ CORREGIR fórmula CV:
        features['cv_flux'] = features['std_flux'] / (features['mean_flux'].abs() + 1e-8)
        
        # ✅ AGREGAR zero crossings:
        features['zero_crossings'] = X.apply(
            lambda row: np.sum(np.diff(np.sign(row - row.mean())) != 0), axis=1
        )
        
        return features

    def create_rolling_features_fixed(self, X):
        """✅ REPLICAR EXACTAMENTE el método del main.py"""
        print("📈 Creating rolling window features (fixed)...")
        
        features = pd.DataFrame(index=X.index)
        
        for window in Config.ROLLING_WINDOWS:
            if window < X.shape[1]:
                print(f"  - Processing window size {window}...")
                
                # ✅ MISMO método manual que main.py
                rolling_means = []
                rolling_stds = []
                rolling_mins = []
                rolling_maxs = []

                for i in range(len(X)):
                    row_values = X.iloc[i].values
                    
                    if len(row_values) >= window:
                        # ✅ MISMA lógica de ventanas múltiples
                        window_stats = []
                        step = max(1, (len(row_values) - window) // 5)  # 5 windows per row
                        
                        for start in range(0, len(row_values) - window + 1, step):
                            window_data = row_values[start:start + window]
                            window_stats.append({
                                'mean': np.mean(window_data),
                                'std': np.std(window_data),
                                'min': np.min(window_data),
                                'max': np.max(window_data)
                            })
                        
                        if window_stats:
                            rolling_means.append(np.mean([w['mean'] for w in window_stats]))
                            rolling_stds.append(np.mean([w['std'] for w in window_stats]))
                            rolling_mins.append(np.min([w['min'] for w in window_stats]))
                            rolling_maxs.append(np.max([w['max'] for w in window_stats]))
                        else:
                            rolling_means.append(np.mean(row_values))
                            rolling_stds.append(np.std(row_values))
                            rolling_mins.append(np.min(row_values))
                            rolling_maxs.append(np.max(row_values))
                    else:
                        rolling_means.append(np.mean(row_values))
                        rolling_stds.append(np.std(row_values))
                        rolling_mins.append(np.min(row_values))
                        rolling_maxs.append(np.max(row_values))

                # ✅ MISMAS características que main.py
                features[f'rolling_mean_{window}'] = rolling_means
                features[f'rolling_std_{window}'] = rolling_stds
                features[f'rolling_min_{window}'] = rolling_mins
                features[f'rolling_max_{window}'] = rolling_maxs
                features[f'rolling_range_{window}'] = np.array(rolling_maxs) - np.array(rolling_mins)

        return features

    def create_frequency_features(self, X):
        """✅ MISMO método que main.py"""
        print("🌊 Creating frequency domain features...")
        
        features = pd.DataFrame(index=X.index)
        
        for i, row in X.iterrows():
            try:
                values = row.values
                fft_values = np.abs(fft(values))
                fft_power = fft_values ** 2  # ✅ AGREGAR que falta

                # Spectral features
                total_power = np.sum(fft_power)  # ✅ Usar fft_power
                if total_power > 0:
                    freq_range = np.arange(len(fft_values))
                    features.loc[i, 'spectral_centroid'] = np.sum(freq_range * fft_power) / total_power
                    features.loc[i, 'spectral_rolloff'] = np.percentile(fft_values, Config.FFT_PERCENTILE)
                    features.loc[i, 'spectral_energy'] = total_power
                    
                    # ✅ AGREGAR high frequency ratio que falta:
                    mid_point = len(fft_values) // 2
                    high_freq_energy = np.sum(fft_power[mid_point:])
                    features.loc[i, 'high_freq_ratio'] = high_freq_energy / total_power
                else:
                    features.loc[i, 'spectral_centroid'] = 0
                    features.loc[i, 'spectral_rolloff'] = 0
                    features.loc[i, 'spectral_energy'] = 0
                    features.loc[i, 'high_freq_ratio'] = 0

                # ✅ CORREGIR nombre de característica:
                peaks, _ = find_peaks(fft_values, height=np.mean(fft_values))
                features.loc[i, 'num_spectral_peaks'] = len(peaks)  # Era 'num_peaks'
                
            except Exception as e:
                # Valores por defecto
                features.loc[i, 'spectral_centroid'] = 0
                features.loc[i, 'spectral_rolloff'] = 0
                features.loc[i, 'spectral_energy'] = 0
                features.loc[i, 'high_freq_ratio'] = 0
                features.loc[i, 'num_spectral_peaks'] = 0
        
        return features

    def create_astronomical_features(self, X):
        """✅ MISMO método que main.py"""
        print("🌌 Creating astronomical features...")
        
        features = pd.DataFrame(index=X.index)
        
        for i, row in X.iterrows():
            try:
                values = row.values
                x_vals = np.arange(len(values))

                # ✅ MISMA lógica de detrending:
                slope, intercept, r_value, _, _ = stats.linregress(x_vals, values)
                detrended = values - (slope * x_vals + intercept)
                
                features.loc[i, 'trend_slope'] = slope
                features.loc[i, 'trend_r_squared'] = r_value ** 2

                # ✅ MISMAS características de tránsito:
                std_thresh = np.std(detrended)
                features.loc[i, 'max_dip'] = np.min(detrended)
                features.loc[i, 'dip_duration'] = np.sum(detrended < -std_thresh)
                features.loc[i, 'num_significant_dips'] = len(find_peaks(-detrended, height=std_thresh)[0])

                # ✅ MISMAS medidas de variabilidad:
                diff_values = np.diff(values)
                features.loc[i, 'total_variation'] = np.sum(np.abs(diff_values))
                features.loc[i, 'max_change'] = np.max(np.abs(diff_values))

                # ✅ MISMA autocorrelación:
                if len(values) > 1:
                    autocorr_1 = np.corrcoef(values[:-1], values[1:])[0, 1]
                    features.loc[i, 'autocorr_1'] = autocorr_1 if not np.isnan(autocorr_1) else 0
                else:
                    features.loc[i, 'autocorr_1'] = 0
                    
            except Exception as e:
                # Valores por defecto
                for col in ['trend_slope', 'trend_r_squared', 'max_dip', 'dip_duration', 
                           'num_significant_dips', 'total_variation', 'max_change', 'autocorr_1']:
                    features.loc[i, col] = 0

        return features

    def create_all_features(self, X):
        """✅ MISMO orden que main.py"""
        print("⚙️ Starting comprehensive feature engineering...")
        
        # ✅ MISMO orden de creación:
        stat_features = self.create_statistical_features(X)
        rolling_features = self.create_rolling_features_fixed(X)  # ✅ Usar versión fixed
        freq_features = self.create_frequency_features(X)
        astro_features = self.create_astronomical_features(X)
        
        # ✅ MISMO orden de concatenación:
        all_features = pd.concat([
            X,  # Original features
            stat_features,
            rolling_features,
            freq_features,
            astro_features
        ], axis=1)
        
        print(f"✅ Feature engineering completed. Total features: {all_features.shape[1]}")
        return all_features

# Después de definir APIDataPreprocessor y APIFeatureEngineer
# Crear aliases para compatibilidad con pickle
DataPreprocessor = APIDataPreprocessor
FeatureEngineer = APIFeatureEngineer

# Agregar al namespace global
import sys
sys.modules[__name__].DataPreprocessor = APIDataPreprocessor
sys.modules[__name__].FeatureEngineer = APIFeatureEngineer

# ✅ ACTUALIZAR líneas 390-400 del modelo:
try:
    # Verificar que el modelo LightGBM existe
    model_path = "model/lightgbm_model.pkl"  # ✅ CAMBIO
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo LightGBM no encontrado en: {model_path}")
    
    predictor = ModeloPredictor(model_path=model_path)
    print(f"✅ Predictor LightGBM inicializado correctamente")
    
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("🔧 Asegúrate de ejecutar el entrenamiento primero:")
    print("   cd ml-entrenamiento && python main.py")
    exit(1)

@app.route('/')
def home():
    """Endpoint raíz para verificar que la API está funcionando"""
    return jsonify({
        "message": "ExoCoders ML API - Predicción con LightGBM",
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
        if df.empty:
            return jsonify({"error": "Empty file", "success": False}), 400
        else:
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
        loggin.error(f"Error procesando archivo: {str(e)}")
        return jsonify({
            "error": f"Error del servidor: {str(e)}", 
            "success": False
        }), 500

@app.route('/health')
def health():
    """Endpoint de salud - solo modelo real"""
    try:
        # Verificar estado del modelo
        model_status = health_check_model("model/lightgbm_model.pkl")
        
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