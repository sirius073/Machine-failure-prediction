from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from lightgbm import LGBMClassifier
import joblib
import os

app = Flask(__name__)

# Constants
MODEL_PATH = os.path.join("model", "best_pipeline.joblib")
DATA_PATH = "data"

# Create directories if they don't exist
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)


@app.route('/upload', methods=['POST'])
def upload_data():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(DATA_PATH, file.filename)
    file.save(file_path)
    return jsonify({"message": "File uploaded successfully", "file_path": file_path})


@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Get the file path from request
        file_path = request.json.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 400

        # Load and validate dataset
        data = pd.read_csv(file_path)
        required_columns = ['Tool wear [min]', 'Torque [Nm]', 'Rotational speed [rpm]', 'Machine failure']
        if not all(col in data.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns: {', '.join(set(required_columns) - set(data.columns))}"}), 400

        # Feature engineering
        data['Power'] = data['Torque [Nm]'] * 2 * np.pi * data['Rotational speed [rpm]'] / 60

        # Define features and labels
        features = ['Tool wear [min]', 'Torque [Nm]', 'Power', 'Rotational speed [rpm]']
        X = data[features]
        y = data['Machine failure']

        # Preprocessing pipeline
        num_pipeline = Pipeline([('std_scaler', StandardScaler())])
        preprocessor_pipeline = ColumnTransformer([('num', num_pipeline, features)])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Model pipeline
        best_estimator = LGBMClassifier(learning_rate=0.1, max_depth=10, n_estimators=100, num_leaves=31)
        best_pipeline = Pipeline([('preprocessor', preprocessor_pipeline), ('model', best_estimator)])
        model = best_pipeline.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Save the model
        joblib.dump(model, MODEL_PATH)

        return jsonify({
            "message": "Model trained successfully",
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc_score": roc_auc
        })
    except FileNotFoundError:
        return jsonify({"error": "File path invalid or file missing"}), 400
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Model not trained"}), 400

        # Load the model
        model = joblib.load(MODEL_PATH)

        # Parse input
        input_data = request.json
        features = ['Tool wear [min]', 'Torque [Nm]', 'Power', 'Rotational speed [rpm]']
        input_values = [input_data.get(feat) for feat in features]

        # Validate inputs
        if None in input_values:
            missing_features = [feat for feat, val in zip(features, input_values) if val is None]
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        # Prepare input for prediction
        input_df = pd.DataFrame([input_values], columns=features)
        prediction = model.predict(input_df)
        confidence = model.predict_proba(input_df)[0].max()

        return jsonify({
            "Machine Failure": "Yes" if prediction[0] == 1 else "No",
            "Confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
