import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model("model.h5")
class_names=['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']


def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    #img_array = np.asarray(img)
    img_array = np.expand_dims(img, axis=0)
    return img_array


def predict_fruit(image_path, model):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    return predicted_class_index

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
      
        predicted_class_index = predict_fruit(file_path, model)
        predicted_class_name = class_names[predicted_class_index]
        result_label.config(text="Predicted Vegetable: " + predicted_class_name)


root = tk.Tk()
root.title("Vegetable Classifier")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()