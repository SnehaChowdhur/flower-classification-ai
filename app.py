
# ==============================
# IMPORTS
# ==============================

import importlib.util
import os
import socket
import subprocess
import sys
import threading
import warnings
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def ensure_dependencies() -> None:
    required_modules = ["tensorflow", "fastapi", "uvicorn", "numpy", "PIL"]
    missing_modules = [
        module for module in required_modules if importlib.util.find_spec(module) is None
    ]

    if not missing_modules:
        return

    requirements_path = BASE_DIR / "requirements.txt"
    if not requirements_path.exists():
        raise RuntimeError(
            "Missing required dependencies: "
            + ", ".join(missing_modules)
            + ". Install them with: pip install "
            + " ".join(missing_modules)
        )

    print(
        "Installing missing project dependencies from requirements.txt...",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
    )


ensure_dependencies()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, HTTPException
# FastAPI creates our backend API

from fastapi.staticfiles import StaticFiles
# Allows FastAPI to serve frontend files such as CSS and JavaScript.

from fastapi.responses import FileResponse
# Allows FastAPI to send index.html to the browser.

import uvicorn

import tensorflow as tf
# Loads the trained TensorFlow model

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

import numpy as np
# Used for arrays and class names

import io
# Used to read uploaded image bytes

from PIL import Image, UnidentifiedImageError
# Used to open and process uploaded images


# ==============================
# CREATE FASTAPI APPLICATION
# ==============================

app = FastAPI()


def find_free_port(start_port: int, max_checks: int = 20) -> int:
    for port in range(start_port, start_port + max_checks):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue

    raise RuntimeError(
        f"Unable to start the API. No free port found near {start_port}."
    )


def run_server() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    start_port = int(os.getenv("PORT", "8000"))
    port = find_free_port(start_port)
    threading.Timer(
        2.0,
        webbrowser.open,
        args=(f"http://{host}:{port}",),
    ).start()

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


# ==============================
# FRONTEND STATIC FILES
# ==============================

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


# ==============================
# LOAD TRAINED MODEL
# ==============================

model = tf.keras.models.load_model(
    str(BASE_DIR / "flower_classifier.keras")
)


# ==============================
# LOAD CLASS NAMES
# ==============================

class_names = np.load(
    str(BASE_DIR / "class_names.npy"),
    allow_pickle=True
)


# ==============================
# HOME PAGE
# ==============================

@app.get("/")
def home():

    return FileResponse(
        str(BASE_DIR / "static" / "index.html")
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

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty. Please choose an image and try again.",
        )


    # ==============================
    # OPEN IMAGE
    # ==============================

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image. Please choose a JPG or PNG image.",
        ) from exc
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


if __name__ == "__main__":
    run_server()
