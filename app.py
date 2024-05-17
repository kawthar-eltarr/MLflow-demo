from flask import Flask, render_template, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

app = Flask(__name__)

# Set the MLflow tracking URI
mlflow_tracking_uri = "http://localhost:8080"
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Experiment name
experiment_name = "WaterQualityPrediction"

# Function to get the latest run's model URI
def get_latest_model_uri():
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return None

    search_results = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"])
    if not search_results.empty:
        latest_run = search_results.iloc[0]
        model_uri = f"runs:/{latest_run.run_id}/random_forest_model"
        return model_uri
    else:
        return None

# Load the model at the start of the Flask app
model_uri = get_latest_model_uri()
if model_uri:
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Model loaded from URI: {model_uri}")
    print(model)
else:
    model = None
    print("No model found in the MLflow tracking server.")


# Generate random data for prediction
random_data = np.random.rand(1, 9)  

# Convert random data to DataFrame
df = pd.DataFrame(random_data)
print(df)

# Make predictions
predictions = model.predict(df)

print(predictions)

@app.route('/')
def home():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    # Generate random data for prediction
    random_data = np.random.rand(1, 9) 

    # Convert random data to DataFrame
    df = pd.DataFrame(random_data,)

    # Make predictions
    predictions = model.predict(df)

    # Display predictions
    return render_template('result.html', predictions=predictions, data=random_data)

if __name__ == '__main__':
    app.run(debug=True)
