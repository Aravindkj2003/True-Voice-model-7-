import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import torch
import torchaudio
import librosa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import DATASET_CONFIG, MODEL_CONFIG
from src.data_loader import MelSpectrogramTransform
from src.model import get_model


ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


app = Flask(__name__)
CORS(app)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _pick_best_model_path() -> Path:
    env_path = os.getenv("MODEL_CHECKPOINT")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    if not DEFAULT_MODEL_DIR.exists():
        raise FileNotFoundError("models directory not found")

    candidates = sorted(DEFAULT_MODEL_DIR.glob("*/best_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        model_url = os.getenv("MODEL_URL")
        if model_url:
            return _download_model_from_url(model_url)

        raise FileNotFoundError(
            "No best_model.pt found under models/*/. Set MODEL_CHECKPOINT or MODEL_URL."
        )

    return candidates[0]


def _download_model_from_url(model_url: str) -> Path:
    # Use temp directory by default so this also works in ephemeral containers.
    cache_dir = Path(os.getenv("MODEL_CACHE_DIR", str(Path(tempfile.gettempdir()) / "deepfake_model")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / "best_model.pt"

    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    response = requests.get(model_url, stream=True, timeout=120)
    response.raise_for_status()

    with target_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    return target_path


def _prepare_input_tensor(audio_path: Path, transform: MelSpectrogramTransform) -> torch.Tensor:
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except Exception:
        # Fallback for environments where compressed formats need extra codecs.
        audio_np, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sr = DATASET_CONFIG["sample_rate"]
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)

    spectrogram = transform(waveform)
    return spectrogram.unsqueeze(0)


class DeepfakeWebService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = _pick_best_model_path()
        self.transform = MelSpectrogramTransform(
            sample_rate=DATASET_CONFIG["sample_rate"],
            n_mels=DATASET_CONFIG["n_mels"],
            n_fft=DATASET_CONFIG["n_fft"],
            hop_length=DATASET_CONFIG["hop_length"],
            spectrogram_size=DATASET_CONFIG["spectrogram_size"],
            normalize_imagenet=True,
        )

        self.model = get_model(
            num_classes=MODEL_CONFIG["num_classes"],
            backbone=MODEL_CONFIG["backbone"],
            pretrained=False,
            device=self.device,
            dropout_rate=MODEL_CONFIG["dropout_rate"],
        )

        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, audio_path: Path) -> dict:
        tensor = _prepare_input_tensor(audio_path, self.transform).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_idx = int(probs.argmax())
        confidence = float(probs[predicted_idx])
        label = "Real" if predicted_idx == 0 else "Fake"

        return {
            "prediction": label,
            "confidence": confidence,
            "scores": {
                "real": float(probs[0]),
                "fake": float(probs[1]),
            },
        }


service = None
service_error = None

try:
    service = DeepfakeWebService()
except Exception as exc:  # pragma: no cover
    service_error = str(exc)


@app.get("/api/health")
def health():
    if service is None:
        return jsonify({"status": "error", "message": service_error}), 500

    return jsonify(
        {
            "status": "ok",
            "device": service.device,
            "model_checkpoint": str(service.model_path),
        }
    )


@app.post("/api/predict")
def predict():
    if service is None:
        return jsonify({"error": service_error}), 500

    if "audio" not in request.files:
        return jsonify({"error": "Missing file field 'audio'"}), 400

    file = request.files["audio"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    tmp_file_path = None
    try:
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = Path(tmp_file.name)

        result = service.predict(tmp_file_path)
        return jsonify(result)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_file_path and tmp_file_path.exists():
            tmp_file_path.unlink(missing_ok=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
