from pathlib import Path
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from src.models.predict_model import predict


app = FastAPI()


# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

prediction_requests = Counter(
    "prediction_requests",
    "Total number of prediction requests",
)

placement_yes = Counter(
    "placement_yes",
    "Total number of Yes predictions",
)

placement_no = Counter(
    "placement_no",
    "Total number of No predictions",
)


# --------------------------------------------------
# Frontend configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class InputData(BaseModel):
    data: dict


# --------------------------------------------------
# Frontend route
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def get_prediction(input: InputData):
    try:
        result = predict(input.data)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # Count every successful prediction
    prediction_requests.inc()

    # Count prediction result
    if str(result).lower() == "yes":
        placement_yes.inc()
    else:
        placement_no.inc()

    # Save prediction information in logs
    logging.info(
        "prediction_request=%s prediction=%s",
        input.data,
        result,
    )

    return {
        "prediction": result
    }


# --------------------------------------------------
# Prometheus metrics endpoint
# --------------------------------------------------

@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )