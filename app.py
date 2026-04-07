import os
import io
import time
import base64
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

import torch
from mlp_architecture import MLP
from data import (get_activation, save_activations)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

OUTPUT_DIR = "outputs"
INPUT_DIR  = "input"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR,  exist_ok=True)

# ── load model ───────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MLP().to(device)
MODEL_PATH = "mlp_mnist.pth"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at '{MODEL_PATH}'. "
        "Please place the .pth file in the project root."
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def _build_mnist_mapping():
    path = r"data/MNIST/raw/mnist-mapping.txt"
    if not os.path.exists(path):
        print(f"Warning: mapping file not found at '{path}'. Using index labels.")
        return {i: str(i) for i in range(10)}
    mapping = {}
    with open(path) as f:
        for line in f:
            key, val = line.split()
            mapping[int(key)] = chr(int(val))
    return mapping

mapping     = _build_mnist_mapping()
activations = {}

hooks = {
    "fc1": model.fc[0],
    "fc2": model.fc[2],
    "fc3": model.fc[4],
}
for name, layer in hooks.items():
    layer.register_forward_hook(get_activation(activations, name))

print(f"Model loaded on {device}  ✓")

# ── last result cache (enables cross-device sync) ─────────────────────────────
last_result: dict = {}


# ── helpers ──────────────────────────────────────────────────────────────────
def preprocess_b64(b64_string: str) -> torch.Tensor:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw  = base64.b64decode(b64_string)
    img  = Image.open(io.BytesIO(raw)).convert("L")
    img  = img.resize((28, 28), Image.LANCZOS)
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = 1.0 - arr
    arr  = arr.reshape(1, 28 * 28)
    scaled = (arr[0] * 255).astype(np.uint8)
    Image.fromarray(scaled).save(os.path.join(INPUT_DIR, "input.png"))
    return torch.from_numpy(arr).float().to(device)


def list_output_images() -> list[dict]:
    images = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.endswith(".png"):
            images.append({
                "filename": fname,
                "label":    fname.replace(".png", "").replace("_", " "),
                "url":      f"/outputs/{fname}",
            })
    return images


# ── routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/viz")
def viz():
    return send_from_directory("templates", "viz.html")


@app.route("/input/<path:filename>")
def serve_input(filename):
    return send_from_directory(INPUT_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    global last_result
    data = request.get_json(force=True)

    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        tensor = preprocess_b64(data["image"])

        with torch.no_grad():
            output = model(tensor)
            pred   = int(torch.argmax(output, dim=1).item())

        char  = mapping.get(pred, "?")
        probs = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()

        label_map = [("1_fc1", "fc1"), ("2_fc2", "fc2"), ("3_fc3", "fc3")]
        for file_label, act_key in label_map:
            if act_key in activations:
                save_activations(activations[act_key], file_label, out_dir=OUTPUT_DIR)

        result = {
            "prediction":   char,
            "label_index":  pred,
            "probabilities": probs,
            "outputs":      list_output_images(),
            "raw_pixels":   tensor.squeeze().tolist(),
            "raw_h1":       activations.get("fc1").squeeze().tolist() if "fc1" in activations else [],
            "raw_h2":       activations.get("fc2").squeeze().tolist() if "fc2" in activations else [],
            "raw_h3":       output.squeeze().tolist(),
            "ts":           time.time(),   # timestamp for change-detection on viz page
        }

        # cache so /latest can serve it to any device
        last_result = result

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/latest")
def latest():
    """Return the most recent prediction result.
    viz.html polls this endpoint so any device stays in sync."""
    return jsonify(last_result)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/outputs")
def list_outputs():
    return jsonify(list_output_images())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)