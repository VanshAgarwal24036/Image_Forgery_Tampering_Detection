import os
import traceback
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# ============================
# CONFIGURATION
# ============================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_FOLDER = os.path.join("static", "uploads")
IMG_SIZE = (224, 224)  # Change based on your model input
CLASS_LABELS = ["Authentic", "Forged"]  # Change if needed
MODEL = None
LAST_ERROR = None

app = Flask(__name__)
app.secret_key = "secure-random-key"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================
# MODEL LOADING
# ============================
def load_model_safely():
    global MODEL, LAST_ERROR
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model.h5 not found at {model_path}")

        # If you have custom layers/metrics, put them here:
        CUSTOM_OBJECTS = {
            # "MyLayer": MyLayer,
            # "f1_score": f1_score,
        }

        MODEL = tf.keras.models.load_model(model_path, compile=False, custom_objects=CUSTOM_OBJECTS)
        try: MODEL.make_predict_function()
        except: pass

        LAST_ERROR = None
        print("✅ Model loaded successfully!")
    except Exception as e:
        import traceback
        LAST_ERROR = f"{e}\n\n{traceback.format_exc()}"
        print("❌ Error loading model:\n", LAST_ERROR)
        MODEL = None


# Load model on startup
load_model_safely()


# ============================
# HELPER FUNCTIONS
# ============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(pil_image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    pil_image = pil_image.resize(IMG_SIZE)
    arr = np.asarray(pil_image).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape → (1, h, w, c)
    return arr


def predict_image(pil_image):
    global MODEL

    if MODEL is None:
        return {"error": LAST_ERROR or "Model not loaded."}

    try:
        batch = preprocess_image(pil_image)
        preds = MODEL.predict(batch)[0]

        # Handle softmax or sigmoid
        if np.isscalar(preds) or preds.shape == ():
            score = float(preds)
            probs = [1 - score, score]
            idx = 1 if score >= 0.5 else 0
        else:
            probs = preds.tolist()
            idx = int(np.argmax(probs))

        label = CLASS_LABELS[idx]
        confidence = float(probs[idx])

        return {
            "label": label,
            "confidence": confidence,
            "probs": probs,
            "error": None
        }

    except Exception as e:
        return {
            "error": f"{e}\n\n{traceback.format_exc()}"
        }


# ============================
# ROUTES
# ============================

@app.route("/")
def index():
    return render_template("index.html", load_error=LAST_ERROR)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No image uploaded.")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Allowed formats: png, jpg, jpeg")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    pil_img = Image.open(save_path)
    result = predict_image(pil_img)

    if result.get("error"):
        flash("Prediction error. See details below.")
        return render_template(
            "index.html",
            filename=filename,
            pred_error=result["error"],
            load_error=LAST_ERROR
        )

    return render_template(
        "index.html",
        filename=filename,
        label=result["label"],
        confidence=f"{result['confidence'] * 100:.2f}",
        load_error=LAST_ERROR
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# API route
@app.route("/api/predict", methods=["POST"])
def api_prediction():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]

    pil_img = Image.open(file.stream)
    result = predict_image(pil_img)

    if result["error"]:
        return jsonify(result), 500

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True)