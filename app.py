from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from agro_model import load_model, load_class_names, load_and_preprocess_image_bytes, predict
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/agro_detect_model.h5"
CLASS_NAMES_PATH = "models/class_names.txt"

model = load_model(MODEL_PATH)
class_names = load_class_names(CLASS_NAMES_PATH)

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    try:
        data = request.get_json()
        if "image" not in data:
            return jsonify({"error": "Falta la clave 'image' en el JSON"}), 400

        # Decodificar la imagen
        image_base64 = data["image"]
        image_bytes = base64.b64decode(image_base64)

        img, _ = load_and_preprocess_image_bytes(image_bytes)
        if img is None:
            return jsonify({"error": "Error al procesar la imagen"}), 500

        results = predict(model, img, class_names)
        if results is None:
            return jsonify({"error": "Error al hacer la predicción"}), 500

        return jsonify({
            "status": "success",
            "top_prediction": results[0],
            "all_predictions": results[:5]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "API funcionando correctamente",
        "timestamp": datetime.now().isoformat()
    })    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
