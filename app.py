# Charger le modèle enregistré avec MLflow
loaded_model = mlflow.sklearn.load_model("random_forest_model")

# Créer une application Flask
from flask import Flask, request, jsonify
app = Flask(__name__)

# Définir une route pour les prédictions
@app.route("/predict", methods=["POST"])
def predict():
    # Récupérer les données d'entrée
    data = request.json
    
    # Effectuer les prédictions avec le modèle chargé
    predictions = loaded_model.predict(data)
    
    # Retourner les prédictions
    return jsonify(predictions)

# Point d'entrée de l'application Flask
if __name__ == "__main__":
    # Démarrer le serveur Flask en mode débogage
    app.run(debug=True)
