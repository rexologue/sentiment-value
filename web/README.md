# Sentiment Classifier Web UI

A minimal FastAPI-powered web UI that proxies requests to the classifier runtime API.

## Run with Docker Compose

```bash
docker compose -f web/docker-compose.yaml up --build
```

Open http://localhost:3000 in your browser.

## Configure the classifier URL

The web app reads the classifier base URL from `CLASSIFIER_API_URL` (default: `http://runtime:8000`).

```bash
CLASSIFIER_API_URL=http://runtime:8000 docker compose -f web/docker-compose.yaml up --build
```

When the classifier runs on the host machine, use:

```bash
CLASSIFIER_API_URL=http://host.docker.internal:8000 docker compose -f web/docker-compose.yaml up --build
```

## Manual test

1. Start the web UI with Docker Compose.
2. Open the page in the browser and confirm the available classes appear automatically.
3. Paste text into the textarea and click **Predict**.
4. Verify the predicted label and class probabilities render in the results card.
