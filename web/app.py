from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

# Make project root importable when running 'python web/app.py'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.SSDMobileNetDetector import SSDMobileNetDetector, BaseDetector
from utils.image_utils   import (
    PrepareModelInput,
    DrawBoundingBoxes,
    ImageArrayToBase64,
    DEFAULT_TARGET_SIZE,
)
from utils.BoundingBox import BoundingBox
from web.PredictionService import PredictionService


WEIGHTS_PATH   = PROJECT_ROOT / "output" / "apple_detector_weights.weights.h5"  # Keras 3 naming
METRICS_PATH   = PROJECT_ROOT / "output" / "metrics.json"
HISTORY_PATH   = PROJECT_ROOT / "output" / "training_history.json"
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.55"))
MAX_UPLOAD_MB  = 16

def CreateApp() -> Flask:
    """
    Application factory pattern (OCP): returns a configured Flask app.
    Tests can call this without starting a server.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    # Instantiate services once at startup (DIP)
    detector           = SSDMobileNetDetector(conf_threshold=CONF_THRESHOLD, nms_iou_threshold=0.15, max_detections=12)
    prediction_service = PredictionService(detector, WEIGHTS_PATH)

    @app.route("/")
    def Index():
        """Serve the main single-page UI."""
        metrics = _LoadJsonSafe(METRICS_PATH)
        history = _LoadJsonSafe(HISTORY_PATH)
        return render_template(
            "index.html",
            model_loaded   = prediction_service.IsModelLoaded,
            metrics        = metrics,
            history        = history,
        )

    @app.route("/predict", methods=["POST"])
    def Predict():
        """
        Accept an image upload and return detection results as JSON.

        Expected: multipart/form-data with field 'image'.
        Returns : JSON — see PredictionService.RunOnPilImage docstring.
        """
        if "image" not in request.files:
            return jsonify({"error": "No image field in request."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        try:
            pil_image = Image.open(io.BytesIO(file.read()))
        except Exception:
            return jsonify({"error": "Cannot decode image."}), 422

        try:
            result = prediction_service.RunOnPilImage(pil_image)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/health")
    def Health():
        """Simple health-check endpoint."""
        return jsonify({
            "status":       "ok",
            "model_loaded": prediction_service.IsModelLoaded,
        })

    return app

def _LoadJsonSafe(path: Path) -> Dict:
    """Load a JSON file; return an empty dict if it does not exist or is invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

if __name__ == "__main__":
    flask_app = CreateApp()
    flask_app.run(host="0.0.0.0", port=5000, debug=False)
