import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import joblib
from PIL import Image, ImageTk

# ============================
# LOAD MODEL
# ============================
model = joblib.load("leaf_model.joblib")
le = joblib.load("label_encoder.joblib")

IMG_SIZE = 128

# ============================
# FUNCTION: PREDICT IMAGE
# ============================
def predict_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if not file_path:
        return

    # Read image
    img = cv2.imread(file_path)

    if img is None:
        result_label.config(text="❌ Error loading image")
        return

    # Resize & preprocess
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features = gray.flatten().reshape(1, -1)

    # Predict
    pred = model.predict(features)
    label = le.inverse_transform(pred)[0]

    # Display result
    result_label.config(text=f"🌱 Prediction: {label}")

    # Show image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((250, 250))

    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk


# ============================
# GUI SETUP
# ============================
root = tk.Tk()
root.title("Leaf Classification System")
root.geometry("400x450")

title = tk.Label(root, text="Leaf Classifier", font=("Arial", 16, "bold"))
title.pack(pady=10)

btn = tk.Button(root, text="Upload Image", command=predict_image, font=("Arial", 12))
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack(pady=10)

result_label = tk.Label(root, text="Prediction will appear here", font=("Arial", 12))
result_label.pack(pady=20)

# ============================
# RUN APP
# ============================
root.mainloop()
