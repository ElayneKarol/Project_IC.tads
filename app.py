# app.py
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import requests

app = Flask(__name__)

# 1. Carrega o modelo treinado (formato .keras)
MODEL_PATH = "mnist_nn_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Rota de predição
@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera um JSON:
    {
      "pixels": [[0, 0, 12, ..., 255],  # 28 valores
                 ...                    # 28 linhas
                ]
    }
    Retorna:
    {
      "predicted_class": 7,
      "probabilities": [0.0001, 0.0023, ..., 0.9102]
    }
    """
    data = request.get_json(force=True)
    pixels = np.array(data["pixels"], dtype=np.float32)  # shape (28,28)
    
    # 3. Pré‑processamento: 
    #    - já vêm em 0–255, igual ao treino do notebook (sem normalização para 0–1)
    #    - converter shape para (1,28,28)
    img = np.expand_dims(pixels, axis=0)

    # 4. Obter os logits e converter para probabilidades
    logits = model.predict(img)            # shape (1,10)
    probs = tf.nn.softmax(logits[0]).numpy()

    # 5. Extrair a classe com maior probabilidade
    pred_class = int(np.argmax(probs))

    return jsonify({
        "predicted_class": pred_class,
        "probabilities": probs.tolist()
    })


if __name__ == "__main__":
    # Para desenvolvimento local
    app.run(host="0.0.0.0", port=5000, debug=True)