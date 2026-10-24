# Verdant AI — Fruit Disease Prediction Frontend

Responsive React + Vite frontend using plain CSS and Lucide icons. It connects to the FastAPI Apple and Mango endpoints:

- `GET /health`
- `POST /predict/mango` (multipart form: image, temperature_c, humidity_pct, rainfall_mm, soil_moisture_pct, soil_ph, season)
- `POST /predict/apple` (same multipart form fields; season defaults to `spring`)

Banana is not available until its model and API route are integrated.

## Run locally

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open the URL printed by Vite (typically `http://localhost:5173`). Keep the FastAPI backend running separately on `http://127.0.0.1:8000`.

The frontend uses `VITE_API_BASE_URL` to configure the backend URL. Do not set a request `Content-Type` for `FormData`; the browser supplies the multipart boundary automatically.
