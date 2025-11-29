from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from runtime.model import ModelWrapper, OptimizationInfo, load_model

LOGGER = logging.getLogger(__name__)
CONFIG_ENV = "SENTIMENT_CONFIG_PATH"
DEFAULT_CONFIG_PATH = "/app/config.yaml"


class ServiceConfig(BaseModel):
    model_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    csv_batch_size: int = 64
    log_level: str = "INFO"
    log_format: Literal["plain", "json"] = "plain"

    @classmethod
    def from_yaml(cls, path: Path) -> "ServiceConfig":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        try:
            return cls(**data)
        except ValidationError as exc:  # pragma: no cover - runtime
            raise ValueError(f"Invalid configuration: {exc}") from exc


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    prediction: int
    negative_score: float
    neutral_score: float
    positive_score: float


class HealthResponse(BaseModel):
    status: str
    device: str
    compiled: bool
    attention: str


def setup_logging(level: str, fmt: Literal["plain", "json"] = "plain") -> None:
    base_format = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        if fmt == "plain"
        else "{\"time\":\"%(asctime)s\",\"level\":\"%(levelname)s\","
        "\"logger\":\"%(name)s\",\"message\":\"%(message)s\"}"
    )

    logging.basicConfig(level=level.upper(), format=base_format)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG_PATH)).expanduser().resolve()
    try:
        config = ServiceConfig.from_yaml(config_path)

    except Exception as exc:  # pragma: no cover - startup failure
        logging.basicConfig(level="INFO")
        LOGGER.exception("Failed to load configuration from %s", config_path)
        raise

    setup_logging(config.log_level, config.log_format)

    LOGGER.info("Using configuration file: %s", config_path)
    LOGGER.info("Configured host=%s port=%s", config.host, config.port)
    LOGGER.info("Configured csv_batch_size=%s", config.csv_batch_size)

    model_path = Path(config.model_path)
    if not model_path.exists():
        LOGGER.error("Model path does not exist: %s", model_path)
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    model = load_model(str(model_path))
    LOGGER.info(
        "Loaded model at %s on device=%s with attention=%s compiled=%s",
        model_path,
        model.device,
        model.info.attn_implementation,
        model.info.compiled,
    )

    app.state.config = config
    app.state.model = model

    yield


def create_app() -> FastAPI:
    return FastAPI(title="Sentiment Inference API", lifespan=lifespan)


app = create_app()


def _probs_to_response(probs: torch.Tensor) -> PredictResponse:
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("Model output must be of shape [batch, 3]")

    scores = probs[0] * 100.0
    prediction = int(torch.argmax(scores).item())

    return PredictResponse(
        prediction=prediction,
        negative_score=round(float(scores[0].item()), 4),
        neutral_score=round(float(scores[1].item()), 4),
        positive_score=round(float(scores[2].item()), 4),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    model: ModelWrapper = app.state.model

    LOGGER.info("/predict called with text length=%d", len(payload.text))

    try:
        probs = model.predict([payload.text], batch_size=1)
    except Exception:
        LOGGER.exception("Model inference failed for /predict")
        raise HTTPException(status_code=500, detail="Model inference failed")

    response = _probs_to_response(probs)
    return response


@app.post("/csv")
async def predict_csv(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    try:
        df = pd.read_csv(file.file)
        
    except Exception:
        LOGGER.exception("Failed to parse uploaded CSV")
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    required_columns = {"ID", "text"}
    if not required_columns.issubset(df.columns):
        raise HTTPException(status_code=400, detail="CSV must contain ID and text columns")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    texts = df["text"].astype(str).tolist()
    ids = df["ID"].tolist()

    model: ModelWrapper = app.state.model
    batch_size = app.state.config.csv_batch_size

    LOGGER.info("/csv called with rows=%d batch_size=%d", len(texts), batch_size)

    predictions: list[float] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            probs = model.predict(batch_texts, batch_size=batch_size)
            batch_preds = torch.argmax(probs, dim=1).tolist()
            predictions.extend(float(p) for p in batch_preds)

    except Exception:
        LOGGER.exception("Model inference failed for /csv")
        raise HTTPException(status_code=500, detail="Model inference failed")

    output_df = pd.DataFrame({"ID": ids, "label": predictions})

    buffer = io.StringIO()
    output_df.to_csv(buffer, index=False)
    buffer.seek(0)

    headers = {
        "Content-Disposition": 'attachment; filename="predictions.csv"',
        "Content-Type": "text/csv",
    }

    return StreamingResponse(buffer, media_type="text/csv", headers=headers)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model: ModelWrapper = app.state.model
    info: OptimizationInfo = model.info

    return HealthResponse(
        status="ok",
        device=str(info.device),
        compiled=info.compiled,
        attention=info.attn_implementation,
    )

if __name__ == "__main__":
    import uvicorn

    config_path = Path(os.environ.get(CONFIG_ENV, DEFAULT_CONFIG_PATH)).expanduser().resolve()
    config = ServiceConfig.from_yaml(config_path)
    setup_logging(config.log_level, config.log_format)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )
