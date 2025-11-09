import tkinter as tk
from tkinter import messagebox, scrolledtext
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- Load Emotion Model ----------
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

CSV_FILE = "results.csv"
KEY_FILE = "secret.key"

# ---------- Encryption Key Setup ----------
def get_key():
    try:
        with open(KEY_FILE, "rb") as f:
            return f.read()
    except FileNotFoundError:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key

key = get_key()
fernet = Fernet(key)

# ---------- Emotion Detection ----------
def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted = emotion_labels[torch.argmax(logits).item()]
    return predicted

# ---------- Encrypt Function ----------
def encrypt_text():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Missing", "Please enter some text.")
        return

    emotion = detect_emotion(text)
    enc = fernet.encrypt(text.encode()).decode()

    df = pd.DataFrame([[text, emotion, enc]], columns=["Text", "Emotion", "Encrypted_Text"])
    try:
        old_df = pd.read_csv(CSV_FILE)
        df = pd.concat([old_df, df], ignore_index=True)
    except:
        pass
    df.to_csv(CSV_FILE, index=False)

    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, f"Detected Emotion: {emotion}\nEncrypted Text:\n{enc}")
    output_box.config(state=tk.DISABLED)

# ---------- Decrypt & Visualize ----------
def decrypt_and_visualize():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        messagebox.showerror("Error", "No results.csv file found yet!")
        return

    decrypted_texts = []
    for enc in df["Encrypted_Text"]:
        try:
            decrypted_texts.append(fernet.decrypt(enc.encode()).decode())
        except:
            decrypted_texts.append("Invalid key or corrupt data")
    df["Decrypted_Text"] = decrypted_texts

    # Show in text box
    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, df.to_string(index=False))
    output_box.config(state=tk.DISABLED)

    # Visualization
    counts = df["Emotion"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(counts.index, counts.values)
    ax.set_title("Emotion Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()

    for widget in chart_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# ---------- GUI ----------
root = tk.Tk()
root.title("Emotion AI Text Encryption")
root.geometry("800x600")
root.config(bg="#f3f3f3")

tk.Label(root, text="Enter Text to Analyze & Encrypt", font=("Arial", 14, "bold"), bg="#f3f3f3").pack(pady=10)
text_input = scrolledtext.ScrolledText(root, height=5, width=80, wrap=tk.WORD, font=("Arial", 12))
text_input.pack(pady=5)

button_frame = tk.Frame(root, bg="#f3f3f3")
button_frame.pack(pady=10)

tk.Button(button_frame, text="🔒 Encrypt Text", command=encrypt_text, width=18, bg="#4CAF50", fg="white",
          font=("Arial", 12, "bold")).grid(row=0, column=0, padx=10)
tk.Button(button_frame, text="🔓 Decrypt & Visualize", command=decrypt_and_visualize, width=22, bg="#2196F3", fg="white",
          font=("Arial", 12, "bold")).grid(row=0, column=1, padx=10)

output_box = scrolledtext.ScrolledText(root, height=10, width=80, wrap=tk.WORD, font=("Arial", 11))
output_box.pack(pady=10)
output_box.config(state=tk.DISABLED)

chart_frame = tk.Frame(root, bg="#f3f3f3")
chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)

root.mainloop()
