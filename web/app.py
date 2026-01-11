import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

DEFAULT_CLASSIFIER_URL = "http://runtime:8000"
DEFAULT_TIMEOUT_SECONDS = 30


def _get_classifier_url() -> str:
    raw = os.getenv("CLASSIFIER_API_URL", DEFAULT_CLASSIFIER_URL)
    return raw.rstrip("/")


def _get_timeout() -> float:
    value = os.getenv("REQUEST_TIMEOUT_SECONDS")
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(value)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


app = FastAPI(title="Sentiment UI")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "classifier_url": _get_classifier_url()}


@app.get("/classes")
async def get_classes() -> Dict[str, Any]:
    url = f"{_get_classifier_url()}/classes"
    timeout = _get_timeout()
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Classifier error: {exc}") from exc
    payload = response.json()
    if "classes" not in payload or not isinstance(payload["classes"], list):
        raise HTTPException(status_code=502, detail="Invalid classifier response")
    return payload


@app.post("/predict")
async def predict(request: PredictRequest) -> Dict[str, Any]:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    url = f"{_get_classifier_url()}/predict"
    timeout = _get_timeout()
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, json={"text": text})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Classifier error: {exc}") from exc
    payload = response.json()
    if "prediction" not in payload:
        raise HTTPException(status_code=502, detail="Invalid classifier response")
    return payload
