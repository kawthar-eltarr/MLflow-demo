import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

# Set the MLflow tracking URI
mlflow_tracking_uri = "http://localhost:8080"
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"MLflow tracking URI set to {mlflow_tracking_uri}")

# load data
df = pd.read_csv("water_potability.csv")

# Set X and y
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Train-validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Ensure the experiment exists
experiment_name = "WaterQualityPrediction"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name)
    print(f"Experiment '{experiment_name}' created.")

experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"Using experiment '{experiment.name}' (ID: {experiment.experiment_id})")

# Start a new run and log parameters, metrics, and model
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    print(f"Started MLflow run with ID: {run.info.run_id}")
    
    # Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    print(f"Model logged with run ID: {run.info.run_id}")

# Ensure the run is finished and committed
print("Run completed and logged successfully.")

# Verify the run is logged by searching for the latest run
search_results = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"])
print("Search Results:")
print(search_results)

if not search_results.empty:
    latest_run = search_results.iloc[0]
    model_uri = f"{mlflow_tracking_uri}/0/{latest_run.run_id}/artifacts/random_forest_model"
    print(f"Model URI: {model_uri}")
else:
    print("No runs found in the MLflow tracking server.")
