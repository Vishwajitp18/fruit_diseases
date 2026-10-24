# Fruit Disease Prediction API

FastAPI backend for the current Apple and Mango prediction models. Banana is
not exposed until its model is integrated.

## Run locally

From the repository root:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive docs are at
`http://127.0.0.1:8000/docs`.

Set `CORS_ORIGINS` to a comma-separated list of frontend origins when using a
different Vite host. The defaults allow `http://localhost:5173` and
`http://localhost:3000`.

## Frontend contract

- `GET /health` reports API availability.
- `POST /predict/apple` predicts Apple disease.
- `POST /predict/mango` predicts Mango disease.

Prediction requests use `multipart/form-data` with `image`,
`temperature_c`, `humidity_pct`, `rainfall_mm`, `soil_moisture_pct`, `soil_ph`,
and `season`. Images are limited to 10 MB. Successful responses include the
fruit, predicted disease, confidence, class probabilities, and submitted
environment values.

## App layout

```text
app/
  api/
    routes/
      health.py
      predictions.py
    router.py
  core/
    config.py
  models/
  schemas/
    prediction.py
  services/
  main.py
tests/
```
