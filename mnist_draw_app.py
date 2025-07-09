import os
import json
import numpy as np
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Configuração da API do back-end
API_URL = os.getenv("API_URL", "http://localhost:5000/predict")

st.set_page_config(page_title="MNIST Draw & Predict", layout="centered")
st.title("Desenhe um número e veja a predição")

# Canvas de desenho
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",  # Fundo transparente
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Se usuário desenhou
if canvas_result.image_data is not None:
    # Converte imagem 280x280 para escala de cinza 28x28
    gray_img = np.mean(canvas_result.image_data[..., :3], axis=2)
    # Redimensiona
    small_img = np.array(
        np.round(
            np.array(
                np.kron(gray_img / 255.0, np.ones((1, 1)))
            ),
        )
    )
    small_img = np.uint8(
        np.array(
            np.round(
                np.array(
                    canvas_result.image_data[..., :3].mean(axis=2)
                )
            )
        )
    )
    # Ajuste correto: usar cv2 (se disponível) para redimensionar:
    try:
        import cv2
        small_img = cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_AREA)
    except ImportError:
        small_img = np.array(
            np.mean(canvas_result.image_data, axis=2)
        )
        small_img = small_img[::10, ::10]

    # Inverte: background branco (255) -> 0, traço preto (0) -> 255
    inverted = (255 - small_img).astype(np.uint8)
    st.subheader("Matriz 28x28"); st.write(inverted.tolist())

    # Botões separados
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download .npy"):
            np.save("drawing.npy", inverted)
            st.success("Arquivo drawing.npy salvo localmente.")
    with col2:
        if st.button("Enviar para predição"):
            # Preprocessar: normalizar conforme treino
            data = inverted.astype(np.float32) / 255.0
            payload = {"pixels": data.tolist()}
            try:
                resp = requests.post(API_URL, json=payload, timeout=5)
                resp.raise_for_status()
                result = resp.json()
                st.success(f"Predição: {result['predicted_class']}")
                st.write(f"Probabilidades: {np.round(result['probabilities'], 3).tolist()}")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao conectar ao servidor: {e}")
