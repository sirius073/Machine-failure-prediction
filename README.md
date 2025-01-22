# Predictive Analysis API for Manufacturing Operations

This project provides a RESTful API for predicting machine failure in manufacturing operations. The API allows you to upload data, train a predictive model, and make predictions using trained models. It is built with Python using Flask and scikit-learn.

---

## Features

- Upload manufacturing data as a CSV file.
- Train a predictive model based on the uploaded data.
- Make predictions about machine failures.

---

## Requirements

1. Python 3.8 or above.
2. Required Python libraries:
   - Flask
   - pandas
   - scikit-learn
   - joblib

---

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install Dependencies:**

   ```bash
   pip install flask pandas scikit-learn joblib
   ```

3. **Project Structure:**

   ```plaintext
   predictive-api/
   ├── app.py                  # Main Flask application
   ├── model/                 # Directory to store trained models
   │   ├── best_pipeline.joblib    # Trained model file (generated after training)
   │   └── machine-failure.ipynb   # Notebook with training on Kaggle dataset
   ├── data/                  # Directory to store uploaded datasets
   │   └── train.csv           # Sample dataset from Kaggle
   ├── README.md              # Documentation
   └── requirements.txt       # List of required dependencies
   ```

4. **Run the Flask Application:**

   ```bash
   python app.py
   ```

   By default, the app runs on `http://127.0.0.1:5000`.

---

## API Endpoints

### 1. **Upload Data**

- **Endpoint:** `POST /upload`
- **Description:** Upload a CSV file containing manufacturing data.
- **Request:**
  ```bash
  curl -X POST -F "file=@E:\ml projects\techpranee\data\train.csv" http://127.0.0.1:5000/upload
  ```
- **Response:**
  ```json
  {
      "message": "File uploaded successfully",
      "file_path": "data/train.csv"
  }
  ```

---

### 2. **Train the Model**

- **Endpoint:** `POST /train`
- **Description:** Train the predictive model using the uploaded dataset.
- **Request:**
  ```bash
  curl -X POST -H "Content-Type: application/json" -d "{\"file_path\": \"data/train.csv\"}" http://127.0.0.1:5000/train
  ```
- **Response:**
  ```json
  {
      "message": "Model trained successfully",
      "accuracy": 0.90,
      "f1_score": 0.75,
      "roc_auc_score": 0.92
  }
  ```

---

### 3. **Make Predictions**

- **Endpoint:** `POST /predict`
- **Description:** Make predictions based on input feature values.
- **Request:**
  ```bash
  curl -X POST -H "Content-Type: application/json" -d "{\"Tool wear [min]\": 120, \"Torque [Nm]\": 50, \"Rotational speed [rpm]\": 1500}" http://127.0.0.1:5000/predict
  ```
- **Response:**
  ```json
  {
      "Machine Failure": "No",
      "Confidence": 0.87
  }
  ```

---

## Example Workflow

1. Upload your dataset using `/upload`.
2. Train the model using `/train`.
3. Use `/predict` to make predictions based on input feature values.

---

## Notes

- Ensure the dataset contains all the required columns (`Tool wear [min]`, `Torque [Nm]`, etc.).
- The trained model is saved as `best_pipeline.joblib` in the `model/` directory.
- Retrain the model if the dataset changes significantly.
- A Jupyter Notebook named `machine-failure.ipynb` is available in the `model/` directory. It contains the model training process on a sample dataset from Kaggle, which can be found [here](https://www.kaggle.com/competitions/machine-failure-prediction-iti-data-science).
- The `train.csv` file used for training is included in the `data/` directory.

---
