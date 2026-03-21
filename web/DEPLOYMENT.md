# Deployment Guide (Render + Vercel)

This guide deploys:
- Backend API (Flask + model inference) on **Render**
- Frontend (Vite + React) on **Vercel**

## 1) Prepare model file hosting

Your model file is currently local and ignored by git (`models/` is in `.gitignore`).
You need a public URL for `best_model.pt`.

Recommended options:
- GitHub Release asset
- Hugging Face model repo file URL
- Google Cloud Storage / S3 public object URL

Model file to upload from local machine:
- `models/20260305_232325/best_model.pt`

## 2) Deploy backend on Render

1. Go to Render dashboard -> `New +` -> `Web Service`.
2. Connect repository: `Aravindkj2003/True-Voice-model-7-`.
3. Configure service:
   - Root Directory: `web/backend`
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 180`
4. Add environment variables:
   - `PORT=10000` (Render default is fine too)
   - `MODEL_URL=<public URL to best_model.pt>`
5. Deploy and wait for build to finish.
6. Test health endpoint:
   - `https://<your-render-service>.onrender.com/api/health`

## 3) Deploy frontend on Vercel

1. Go to Vercel -> `Add New...` -> `Project`.
2. Import same GitHub repo.
3. Set project settings:
   - Framework Preset: `Vite`
   - Root Directory: `web/frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. Add environment variable:
   - `VITE_API_BASE_URL=https://<your-render-service>.onrender.com`
5. Deploy.

## 4) Verify end-to-end

1. Open frontend URL from Vercel.
2. Click `Check Backend`.
3. Upload a small audio file and run analysis.

## Notes

- First backend request may be slower if Render instance is sleeping.
- Model download happens automatically on backend startup when `MODEL_URL` is set.
- If model loading fails, check Render logs for URL/access errors.
