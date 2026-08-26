
# ==============================
# IMPORTS
# ==============================

from fastapi import FastAPI, UploadFile, File
# FastAPI creates our backend API

from fastapi.staticfiles import StaticFiles
# ⭐ CHANGED: Added this import.
# What changed:
# Allows FastAPI to serve frontend files such as CSS and JavaScript.

from fastapi.responses import FileResponse
# ⭐ CHANGED: Added this import.
# What changed:
# Allows FastAPI to send index.html to the browser.

import tensorflow as tf
# Loads the trained TensorFlow model

import numpy as np
# Used for arrays and class names

import io
# Used to read uploaded image bytes

from PIL import Image
# Used to open and process uploaded images


# ==============================
# CREATE FASTAPI APPLICATION
# ==============================

app = FastAPI()


# ==============================
# FRONTEND STATIC FILES
# ==============================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
# ⭐ CHANGED: Added this entire section.
#
# What actually changed:
# FastAPI is now connected to the "static" folder.
#
# This allows the browser to access:
#
# /static/index.html
# /static/style.css
# /static/script.js


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = tf.keras.models.load_model(
    "flower_classifier.keras"
)
# Existing code:
# Loads the trained flower classification model


# ==============================
# LOAD CLASS NAMES
# ==============================

class_names = np.load(
    "class_names.npy",
    allow_pickle=True
)
# Existing code:
# Loads the 13 flower names


# ==============================
# HOME PAGE
# ==============================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )

# ⭐ CHANGED: Added this endpoint.
#
# What actually changed:
# Previously "/" returned:
#
# {"message": "Flower Classification API is running"}
#
# Now "/" returns the actual website:
#
# static/index.html
#
# Therefore:
# http://127.0.0.1:8000
#
# opens our frontend.


# ==============================
# PREDICTION ENDPOINT
# ==============================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Existing endpoint:
    # Receives the image from the frontend


    # ==============================
    # READ IMAGE
    # ==============================

    image_bytes = await file.read()
    # Read the uploaded image


    # ==============================
    # OPEN IMAGE
    # ==============================

    image = Image.open(
        io.BytesIO(image_bytes)
    )
    # Convert uploaded bytes into an image


    # ==============================
    # CONVERT TO RGB
    # ==============================

    image = image.convert(
        "RGB"
    )
    # Make sure the image has 3 colour channels


    # ==============================
    # RESIZE IMAGE
    # ==============================

    image = image.resize(
        (224, 224)
    )
    # Resize to the same size used during training


    # ==============================
    # CONVERT TO NUMPY ARRAY
    # ==============================

    image_array = np.array(image)
    # Convert image into a NumPy array


    # ==============================
    # NORMALIZE IMAGE
    # ==============================

    image_array = image_array.astype(
        "float32"
    ) / 255.0
    # Convert pixel values from 0-255 to 0-1


    # ==============================
    # ADD BATCH DIMENSION
    # ==============================

    image_array = np.expand_dims(
        image_array,
        axis=0
    )
    # Add batch dimension required by TensorFlow


    # ==============================
    # PREDICT
    # ==============================

    predictions = model.predict(
        image_array
    )
    # Ask the trained model to classify the flower


    # ==============================
    # FIND PREDICTED CLASS
    # ==============================

    predicted_index = np.argmax(
        predictions[0]
    )
    # Find the class with the highest probability


    # ==============================
    # GET FLOWER NAME
    # ==============================

    predicted_flower = class_names[
        predicted_index
    ]
    # Convert class number into flower name


    # ==============================
    # GET CONFIDENCE
    # ==============================

    confidence = float(
        predictions[0][predicted_index]
    )
    # Get the model's confidence


    # ==============================
    # SEND RESULT TO FRONTEND
    # ==============================

    return {
        "flower": str(predicted_flower),
        "confidence": round(
            confidence * 100,
            2
        )
    }

    # Existing prediction response.
    # The frontend receives:
    #
    # {
    #     "flower": "Golap",
    #     "confidence": 95.42
    # }
