import os
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

# Configurações via variáveis de ambiente
MODEL_PATH = os.getenv("MODEL_PATH", "mnist_nn_model.keras")
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", 5000))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

# Setup do Flask
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Carrega o modelo com custom_objects para desserializar softmax_v2
try:
    custom_objs = {"softmax_v2": tf.keras.activations.softmax}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objs)
    logging.info(f"Modelo carregado de: {MODEL_PATH}")
except Exception as e:
    logging.error(f"Falha ao carregar modelo: {e}")
    raise

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if "pixels" not in data:
            return jsonify({"error": "Chave 'pixels' ausente"}), 400

        # Converte diretamente para float32 sem normalização
        pixels = np.array(data["pixels"], dtype=np.float32)
        if pixels.shape != (28, 28):
            return jsonify({"error": f"Formato inválido: esperado (28,28), recebido {pixels.shape}"}), 400

        # Expande dims: canal e batch para shape (1,28,28,1)
        img = np.expand_dims(pixels, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Predição
        preds = model.predict(img)
        # Se a saída já for probabilidades (softmax), usa direto
        probs = preds[0].tolist()
        pred_class = int(np.argmax(probs))

        return jsonify({
            "predicted_class": pred_class,
            "probabilities": probs
        })
    except Exception as e:
        logging.exception("Erro no endpoint /predict")
        return jsonify({"error": "Erro interno do servidor", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)