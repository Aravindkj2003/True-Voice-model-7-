# Deepfake Audio Detector – Web Application

## Quick Start

You have a working **Flask** backend API and **React** frontend ready to run.

### Option 1: Run Both Servers (Recommended)

**Terminal 1 – Start Flask Backend (port 5000):**
```bash
cd "C:\Users\Aravind KJ\Desktop\deepfake 7\web\backend"
python app.py
```

**Terminal 2 – Start React Frontend (port 5173):**
```bash
cd "C:\Users\Aravind KJ\Desktop\deepfake 7\web\frontend"
npm run dev
```

Then open your browser:
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:5000

### Option 2: Run Just the Backend (API-only)

```bash
cd "C:\Users\Aravind KJ\Desktop\deepfake 7\web\backend"
python app.py
```

Test the API with curl:
```bash
# Check health
curl http://localhost:5000/api/health

# Upload an audio file
curl -X POST -F "audio=@path/to/audio.wav" http://localhost:5000/api/predict
```

---

## Project Structure

```
web/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt     # Python dependencies
│   └── .env.example         # Optional model path override
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── main.jsx         # React entry point
│   │   └── styles.css       # Styling
│   ├── package.json         # Node dependencies
│   ├── vite.config.js       # Vite configuration
│   ├── index.html           # HTML entry
│   └── .env.example         # API URL config
```

---

## Backend Endpoints

### GET /api/health
Check if the backend is running and model is loaded.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "model_checkpoint": "C:\\...\\best_model.pt"
}
```

### POST /api/predict
Upload an audio file and get a deepfake prediction.

**Request:**
- `audio` (file): WAV, MP3, FLAC, OGG, or M4A

**Response:**
```json
{
  "prediction": "Real",
  "confidence": 0.9735,
  "scores": {
    "real": 0.9735,
    "fake": 0.0265
  }
}
```

---

## Frontend Features

- **File Upload:** Select audio file (drag & drop supported)
- **Real-time Health Check:** Verify backend connectivity
- **Live Predictions:** See confidence scores and classification
- **Responsive Design:** Works on desktop and mobile

---

## Configuration

### Backend (.env)
```
PORT=5000                    # Flask server port
MODEL_CHECKPOINT=...         # Optional: override auto-discovered model
```

### Frontend (.env)
```
VITE_API_BASE_URL=http://localhost:5000
```

---

## Performance Notes

- Model: ResNet-18 (trained on your 56K dataset)
- GPU: NVIDIA RTX 3050 (inference is fast)
- Supported formats: WAV, MP3, FLAC, OGG, M4A
- Max training accuracy: **99.96%**
- Best validation accuracy: **97.33%** (epoch 46)

---

## Troubleshooting

**Backend won't start:**
- Ensure Python 3.8+ is available
- Check that the models directory exists and contains a checkpoint
- Verify port 5000 is not in use

**Frontend won't connect to backend:**
- Ensure both servers are running
- Check that `VITE_API_BASE_URL` matches your backend address
- Verify CORS is not blocked (already configured in `app.py`)

**File upload fails:**
- Check file size (no limit set, but should be < 500 MB)
- Verify file format is supported
- Check backend logs for errors

---

## Next Steps

1. Run both servers (see Quick Start)
2. Open http://localhost:5173 in your browser
3. Click "Check Backend" to verify connectivity
4. Upload an audio file and click "Analyze Audio"
5. View the prediction result with confidence score

---

Enjoy your deepfake audio detection web app! 🎵🔍
