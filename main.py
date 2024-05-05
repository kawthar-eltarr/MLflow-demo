import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Setup MLflow server on port 5000
mlflow.set_tracking_uri("http://localhost:5000")

# load data
df = pd.read_csv("water_potability.csv")

# Set X and y
X = df.drop(columns=["Potability"])
y = df["Potability"]

# Train-validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser et entraîner le modèle Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Calculer l'exactitude du modèle
accuracy = accuracy_score(y_test, y_pred)
print("\n")
print("Accuracy :", accuracy)
print("\n")
print("Classification report :", classification_report(y_test, y_pred))


# Si l'expérience n'existe pas, la créer
experiment_name = "WaterQualityPrediction"

if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name) 

experiment = mlflow.get_experiment_by_name(experiment_name)

# Start up the experiment
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")


loaded_model = mlflow.sklearn.load_model("random_forest_model")

