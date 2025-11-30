from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from functools import lru_cache

import pandas as pd
import joblib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Segmenter,
)

from model import ModelWrapper, OptimizationInfo, load_model
from sentiment_value.clustering.normalize_and_pca import l2_normalize

LOGGER = logging.getLogger(__name__)
CONFIG_ENV = "SENTIMENT_CONFIG_PATH"
DEFAULT_CONFIG_PATH = "/app/config.yaml"


class ServiceConfig(BaseModel):
    model_path: str
    prefer_cuda: bool = True
    enable_compile: bool = False
    cluster_model_path: Optional[str] = None
    cluster_pca_path: Optional[str] = None
    cluster_label_map_path: Optional[str] = None
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


class ClusterPredictResponse(BaseModel):
    class_id: int
    cluster_id: int
    distance: Optional[float] = None


class LemmatizeResponse(BaseModel):
    output: str


class NerResponse(BaseModel):
    entities: list[str]


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

    model = load_model(
        str(model_path),
        prefer_cuda=config.prefer_cuda,
        enable_compile=config.enable_compile,
    )
    LOGGER.info(
        "Loaded model at %s on device=%s with attention=%s compiled=%s",
        model_path,
        model.device,
        model.info.attn_implementation,
        model.info.compiled,
    )

    cluster_model = None
    cluster_pca = None
    cluster_label_map: Optional[dict[int, int]] = None
    if config.cluster_model_path:
        cluster_model_path = Path(config.cluster_model_path).expanduser().resolve()
        if not cluster_model_path.exists():
            raise FileNotFoundError(
                f"Clustering model path does not exist: {cluster_model_path}"
            )
        cluster_model = joblib.load(cluster_model_path)
        LOGGER.info("Loaded clustering model from %s", cluster_model_path)

        if config.cluster_pca_path:
            cluster_pca_path = Path(config.cluster_pca_path).expanduser().resolve()
            if not cluster_pca_path.exists():
                raise FileNotFoundError(
                    f"PCA model path does not exist: {cluster_pca_path}"
                )
            cluster_pca = joblib.load(cluster_pca_path)
            LOGGER.info("Loaded clustering PCA transformer from %s", cluster_pca_path)

        if config.cluster_label_map_path:
            label_map_path = Path(config.cluster_label_map_path).expanduser().resolve()
            if not label_map_path.exists():
                raise FileNotFoundError(
                    f"Cluster label map path does not exist: {label_map_path}"
                )

            with label_map_path.open("r", encoding="utf-8") as f:
                loaded_map = yaml.safe_load(f)

            if not isinstance(loaded_map, dict) or not loaded_map:
                raise ValueError(
                    "Cluster label map must be a non-empty mapping of cluster_id to class_id"
                )

            cluster_label_map = {}
            for key, value in loaded_map.items():
                try:
                    cluster_key = int(key)
                    class_value = int(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        "Cluster label map keys and values must be integers"
                    ) from None

                if class_value not in (0, 1, 2):
                    raise ValueError(
                        f"Cluster label map contains invalid class {class_value}; expected 0, 1 or 2"
                    )

                cluster_label_map[cluster_key] = class_value

            LOGGER.info(
                "Loaded cluster label map with %s entries from %s",
                len(cluster_label_map),
                label_map_path,
            )

    app.state.config = config
    app.state.model = model
    app.state.cluster_model = cluster_model
    app.state.cluster_pca = cluster_pca
    app.state.cluster_label_map = cluster_label_map

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Sentiment Inference API", lifespan=lifespan)

    # Максимально открытый CORS: любой origin, любые методы и заголовки
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


@lru_cache(maxsize=1)
def _load_ru_nlp_pipeline():
    """Lazy-load lightweight Russian NLP pipeline for lemmatization and NER."""

    segmenter = Segmenter()
    embeddings = NewsEmbedding()
    morph_vocab = MorphVocab()
    morph_tagger = NewsMorphTagger(embeddings)
    ner_tagger = NewsNERTagger(embeddings)

    return segmenter, morph_vocab, morph_tagger, ner_tagger


def _lemmatize_text(text: str) -> str:
    segmenter, morph_vocab, morph_tagger, _ = _load_ru_nlp_pipeline()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    lemmas = [token.lemma for token in doc.tokens if token.lemma]
    return " ".join(lemmas)


def _extract_entities(text: str) -> list[str]:
    segmenter, _, _, ner_tagger = _load_ru_nlp_pipeline()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    return [span.text for span in doc.spans]


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


def _prepare_cluster_features(
    embeddings: torch.Tensor, pca_model: Optional[object]
) -> np.ndarray:
    """Convert embeddings to numpy features for clustering.

    The vectors are L2-normalized to mirror ``normalize_and_pca.py`` and
    optionally passed through a PCA transformer if provided.
    """

    vectors = embeddings.detach().cpu().numpy().astype("float32")
    vectors = l2_normalize(vectors, np)

    if pca_model is not None:
        vectors = pca_model.transform(vectors)

    return vectors


def _predict_labels(model: ModelWrapper, texts: list[str], batch_size: int) -> list[float]:
    predictions: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        probs = model.predict(batch_texts, batch_size=batch_size)
        batch_preds = torch.argmax(probs, dim=1).tolist()
        predictions.extend(float(p) for p in batch_preds)

    return predictions


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    model: ModelWrapper = app.state.model

    LOGGER.info("/predict called with text length=%d", len(payload.text))

    try:
        probs = await run_in_threadpool(model.predict, [payload.text], batch_size=1)
    except Exception:
        LOGGER.exception("Model inference failed for /predict")
        raise HTTPException(status_code=500, detail="Model inference failed")

    response = _probs_to_response(probs)
    return response


@app.post("/cluster", response_model=ClusterPredictResponse)
async def cluster(payload: PredictRequest) -> ClusterPredictResponse:
    model: ModelWrapper = app.state.model
    cluster_model = getattr(app.state, "cluster_model", None)
    cluster_pca = getattr(app.state, "cluster_pca", None)
    cluster_label_map = getattr(app.state, "cluster_label_map", None)

    if cluster_model is None:
        raise HTTPException(status_code=503, detail="Clustering model is not configured")

    if cluster_label_map is None:
        raise HTTPException(
            status_code=503,
            detail="Cluster label map is not configured; cannot produce class predictions",
        )

    LOGGER.info(
        "/cluster called with text length=%d pca=%s",
        len(payload.text),
        cluster_pca is not None,
    )

    try:
        embeddings = await run_in_threadpool(
            model.cls_embeddings, [payload.text], batch_size=1
        )
        features = await run_in_threadpool(
            _prepare_cluster_features, embeddings, cluster_pca
        )
        preds = await run_in_threadpool(cluster_model.predict, features)
        cluster_id = int(preds[0])

        if cluster_id not in cluster_label_map:
            raise HTTPException(
                status_code=422,
                detail=f"Cluster id {cluster_id} is missing in the configured label map",
            )
        class_id = int(cluster_label_map[cluster_id])

        distance: Optional[float] = None
        if hasattr(cluster_model, "transform"):
            distances = await run_in_threadpool(cluster_model.transform, features)
            if distances.size:
                distance = float(np.min(distances[0]))

    except HTTPException:
        raise
    except Exception:
        LOGGER.exception("Clustering inference failed for /cluster")
        raise HTTPException(status_code=500, detail="Clustering inference failed")

    return ClusterPredictResponse(
        class_id=class_id, cluster_id=cluster_id, distance=distance
    )


@app.post("/lemmatize", response_model=LemmatizeResponse)
async def lemmatize(payload: PredictRequest) -> LemmatizeResponse:
    LOGGER.info("/lemmatize called with text length=%d", len(payload.text))

    try:
        output = await run_in_threadpool(_lemmatize_text, payload.text)
    except Exception:
        LOGGER.exception("Lemmatization failed for /lemmatize")
        raise HTTPException(status_code=500, detail="Lemmatization failed")

    return LemmatizeResponse(output=output)


@app.post("/ner", response_model=NerResponse)
async def ner(payload: PredictRequest) -> NerResponse:
    LOGGER.info("/ner called with text length=%d", len(payload.text))

    try:
        entities = await run_in_threadpool(_extract_entities, payload.text)
    except Exception:
        LOGGER.exception("NER failed for /ner")
        raise HTTPException(status_code=500, detail="Entity extraction failed")

    return NerResponse(entities=entities)


@app.post("/csv")
async def predict_csv(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    try:
        df = await run_in_threadpool(pd.read_csv, file.file)
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

    try:
        predictions = await run_in_threadpool(
            _predict_labels, model, texts, batch_size
        )
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
