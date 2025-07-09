import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image


st.title("🖍️ Desenhe um Dígito (28x28 pixels)")

# Configurações da tela de desenho
canvas_result = st_canvas(
    fill_color="white",  # Cor de fundo
    stroke_width=10,      # Espessura do traço
    stroke_color="black", # Cor do traço
    background_color="white",
    width=280,            # Tamanho em pixels (10x a resolução final)
    height=280,
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

    if st.button("Salvar como .npy"):
        np.save("desenho_28x28.npy", img_array)
        st.success("Imagem salva como 'desenho_28x28.npy'")

# img_array é seu array 28×28 de 0–255
        payload = {"pixels": img_array.tolist()}
        resp = requests.post("http://localhost:5000/predict", json=payload)
        result = resp.json()

        st.subheader("Predição do Modelo")
        st.write(f"🖥 Classe: {result['predicted_class']}")
        st.write("📊 Probabilidades:")
        st.write(result["probabilities"])