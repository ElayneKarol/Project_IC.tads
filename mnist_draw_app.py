import os
import json
import numpy as np
import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

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

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data).astype("uint8"))
    img_resized = img.resize((28, 28)).convert("L")

    st.subheader("🖼️ Visualização da Imagem 28x28")
    st.image(img_resized, width=150)

    img_array = np.array(img_resized)
    img_array = 255 - img_array  # inverte: fundo branco vira 0, traço preto vira 255

    st.subheader("📊 Matriz de pixels (28x28) — escala de 0 a 255")
    st.write(img_array)


    # Botões separados
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download .npy"):
            np.save("drawing.npy", inverted)
            st.success("Arquivo drawing.npy salvo localmente.")
    with col2:
        if st.button("Enviar para predição", key="predict"):
            payload = {"pixels": img_array.astype(np.float32).tolist()}
            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                response.raise_for_status()
                result = response.json()
                pred = result['predicted_class']
                probs = result['probabilities']
                # Converte para porcentagem e filtra >0
                percents = [round(p * 100) for p in probs]
                items = [(i, percents[i]) for i in range(len(percents)) if percents[i] > 0]
                # Ordena por probabilidade
                items.sort(key=lambda x: x[1], reverse=True)
                # Formata como "digit=xx%"
                formatted = ", ".join(f"{digit}={pct}%" for digit, pct in items)

                st.success(f"Valor Predito: {pred}")
                st.write(f"Probabilidades: {formatted}")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao conectar ao servidor: {e}")
