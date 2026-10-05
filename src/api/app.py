from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from src.training.train_baseline import CLASS_NAMES, build_model, build_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_resnet18.pt"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "baseline" / "best_baseline.json"

UI_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mushroom Classifier</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f3;
      --panel: #ffffff;
      --text: #1d2520;
      --muted: #647067;
      --line: #dbe0d8;
      --accent: #2f7d57;
      --accent-dark: #215d42;
      --warn: #9a5b11;
      --danger: #b42318;
      --shadow: 0 12px 36px rgba(31, 41, 55, 0.10);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }

    .topbar {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.15;
      font-weight: 760;
      letter-spacing: 0;
    }

    .subtitle {
      color: var(--muted);
      font-size: 13px;
    }

    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
      gap: 18px;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }

    .section-head {
      min-height: 56px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    h2 {
      margin: 0;
      font-size: 16px;
      font-weight: 720;
      letter-spacing: 0;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 28px;
      padding: 0 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--warn);
    }

    .dot.ok { background: var(--accent); }
    .dot.bad { background: var(--danger); }

    .content {
      padding: 18px;
    }

    .dropzone {
      border: 1px dashed #9ca99f;
      border-radius: 8px;
      min-height: 260px;
      display: grid;
      place-items: center;
      background: #fbfcfa;
      overflow: hidden;
      position: relative;
    }

    .dropzone input {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }

    .dropzone img {
      width: 100%;
      height: 100%;
      max-height: 360px;
      object-fit: contain;
      display: none;
      background: #f1f3ef;
    }

    .empty {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      text-align: center;
      padding: 20px;
    }

    .empty strong {
      color: var(--text);
      font-size: 16px;
    }

    .actions {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      flex-wrap: wrap;
    }

    button {
      min-height: 40px;
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 0 14px;
      background: var(--accent);
      color: #fff;
      font-weight: 680;
      cursor: pointer;
    }

    button:hover { background: var(--accent-dark); }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }

    .secondary {
      background: #ffffff;
      color: var(--text);
      border-color: var(--line);
    }

    .secondary:hover {
      background: #f3f5f1;
    }

    .result {
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 16px;
      display: grid;
      gap: 14px;
    }

    .prediction {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }

    .label {
      font-size: 30px;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: none;
    }

    .confidence {
      color: var(--muted);
      font-size: 14px;
    }

    .bars {
      display: grid;
      gap: 10px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 98px minmax(0, 1fr) 56px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }

    .bar {
      height: 10px;
      border-radius: 999px;
      background: #e7ebe4;
      overflow: hidden;
    }

    .fill {
      height: 100%;
      width: 0%;
      background: var(--accent);
      border-radius: inherit;
      transition: width 180ms ease;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 76px;
      background: #fbfcfa;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }

    .metric strong {
      font-size: 20px;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }

    .subhead {
      margin: 18px 0 10px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    pre {
      margin: 0;
      padding: 12px;
      border-radius: 8px;
      background: #17211b;
      color: #e9f1ea;
      overflow: auto;
      max-height: 240px;
      font-size: 12px;
      line-height: 1.45;
    }

    .error {
      color: var(--danger);
      font-size: 14px;
      min-height: 20px;
    }

    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; padding: 14px 0; flex-direction: column; }
      .bar-row { grid-template-columns: 86px minmax(0, 1fr) 48px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <h1>Mushroom Classifier</h1>
        <div class="subtitle">проверка изображения: edible / non_edible</div>
      </div>
      <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">загрузка</span></div>
    </div>
  </header>

  <main>
    <section>
      <div class="section-head">
        <h2>Предсказание</h2>
        <button id="predictButton" disabled>Проверить</button>
      </div>
      <div class="content">
        <label class="dropzone">
          <input id="fileInput" type="file" accept="image/*">
          <img id="preview" alt="Selected mushroom image">
          <div id="emptyState" class="empty">
            <strong>Выберите изображение</strong>
            <span>JPG, JPEG, PNG, WEBP, BMP</span>
          </div>
        </label>
        <div class="actions">
          <button id="clearButton" class="secondary" type="button">Очистить</button>
          <button id="refreshButton" class="secondary" type="button">Обновить мониторинг</button>
        </div>
        <div id="error" class="error"></div>
        <div id="result" class="result" style="display:none">
          <div class="prediction">
            <div id="predictedLabel" class="label">-</div>
            <div id="confidence" class="confidence">уверенность -</div>
          </div>
          <div class="bars">
            <div class="bar-row">
              <span>edible</span>
              <div class="bar"><div id="edibleBar" class="fill"></div></div>
              <strong id="edibleValue">0%</strong>
            </div>
            <div class="bar-row">
              <span>non_edible</span>
              <div class="bar"><div id="nonEdibleBar" class="fill"></div></div>
              <strong id="nonEdibleValue">0%</strong>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>Мониторинг</h2>
        <span id="modelBadge" class="status"><span class="dot"></span><span>модель</span></span>
      </div>
      <div class="content">
        <div class="grid">
          <div class="metric"><span>запросы</span><strong id="requests">0</strong></div>
          <div class="metric"><span>ошибки</span><strong id="errors">0</strong></div>
          <div class="metric"><span>средняя latency</span><strong id="latency">0 ms</strong></div>
          <div class="metric"><span>память</span><strong id="memory">0%</strong></div>
          <div class="metric"><span>cpu</span><strong id="cpu">0%</strong></div>
          <div class="metric"><span>device</span><strong id="device">-</strong></div>
        </div>
        <div class="subhead">Качество baseline</div>
        <div class="grid">
          <div class="metric"><span>accuracy</span><strong id="qualityAccuracy">-</strong></div>
          <div class="metric"><span>precision</span><strong id="qualityPrecision">-</strong></div>
          <div class="metric"><span>recall</span><strong id="qualityRecall">-</strong></div>
          <div class="metric"><span>f1</span><strong id="qualityF1">-</strong></div>
          <div class="metric"><span>roc-auc</span><strong id="qualityRocAuc">-</strong></div>
          <div class="metric"><span>split</span><strong id="qualitySplit">-</strong></div>
        </div>
        <div style="height:14px"></div>
        <pre id="rawMonitoring">{}</pre>
      </div>
    </section>
  </main>

  <script>
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const emptyState = document.getElementById("emptyState");
    const predictButton = document.getElementById("predictButton");
    const clearButton = document.getElementById("clearButton");
    const refreshButton = document.getElementById("refreshButton");
    const result = document.getElementById("result");
    const errorBox = document.getElementById("error");

    let selectedFile = null;

    function pct(value) {
      return `${Math.round(Number(value) * 1000) / 10}%`;
    }

    function metricPct(value) {
      return value === null || value === undefined ? "-" : pct(value);
    }

    function setError(message) {
      errorBox.textContent = message || "";
    }

    function setProbability(idPrefix, value) {
      document.getElementById(`${idPrefix}Bar`).style.width = pct(value);
      document.getElementById(`${idPrefix}Value`).textContent = pct(value);
    }

    fileInput.addEventListener("change", () => {
      selectedFile = fileInput.files[0] || null;
      setError("");
      result.style.display = "none";
      predictButton.disabled = !selectedFile;

      if (!selectedFile) return;
      const url = URL.createObjectURL(selectedFile);
      preview.src = url;
      preview.style.display = "block";
      emptyState.style.display = "none";
    });

    clearButton.addEventListener("click", () => {
      selectedFile = null;
      fileInput.value = "";
      preview.removeAttribute("src");
      preview.style.display = "none";
      emptyState.style.display = "flex";
      predictButton.disabled = true;
      result.style.display = "none";
      setError("");
    });

    predictButton.addEventListener("click", async () => {
      if (!selectedFile) return;
      setError("");
      predictButton.disabled = true;

      const body = new FormData();
      body.append("file", selectedFile);

      try {
        const response = await fetch("/predict", { method: "POST", body });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Не удалось получить предсказание");

        document.getElementById("predictedLabel").textContent = payload.predicted_class;
        document.getElementById("confidence").textContent = `уверенность ${pct(payload.confidence)}`;
        setProbability("edible", payload.probabilities.edible);
        setProbability("nonEdible", payload.probabilities.non_edible);
        result.style.display = "grid";
        await loadMonitoring();
      } catch (error) {
        setError(error.message);
      } finally {
        predictButton.disabled = !selectedFile;
      }
    });

    async function loadMonitoring() {
      const response = await fetch("/monitoring");
      const payload = await response.json();

      const statusDot = document.getElementById("statusDot");
      const statusText = document.getElementById("statusText");
      const modelBadge = document.getElementById("modelBadge");
      const modelDot = modelBadge.querySelector(".dot");

      statusText.textContent = payload.service.status === "ok" ? "сервис готов" : "сервис ограничен";
      statusDot.className = `dot ${payload.service.status === "ok" ? "ok" : "bad"}`;
      modelBadge.querySelector("span:last-child").textContent = payload.model.loaded ? "модель загружена" : "модель недоступна";
      modelDot.className = `dot ${payload.model.loaded ? "ok" : "bad"}`;

      document.getElementById("requests").textContent = payload.prediction_metrics.requests_total;
      document.getElementById("errors").textContent = payload.prediction_metrics.errors_total;
      document.getElementById("latency").textContent =
        `${Math.round(payload.prediction_metrics.average_latency_seconds * 1000)} ms`;
      document.getElementById("memory").textContent = `${Math.round(payload.infrastructure.memory_percent)}%`;
      document.getElementById("cpu").textContent = `${Math.round(payload.infrastructure.cpu_percent)}%`;
      document.getElementById("device").textContent = payload.infrastructure.device;

      const metrics = payload.metrics || {};
      document.getElementById("qualityAccuracy").textContent = metricPct(metrics.accuracy);
      document.getElementById("qualityPrecision").textContent = metricPct(metrics.precision);
      document.getElementById("qualityRecall").textContent = metricPct(metrics.recall);
      document.getElementById("qualityF1").textContent = metricPct(metrics.f1);
      document.getElementById("qualityRocAuc").textContent = metricPct(metrics.roc_auc);
      document.getElementById("qualitySplit").textContent = metrics.available ? metrics.split : "-";
      document.getElementById("rawMonitoring").textContent = JSON.stringify(payload, null, 2);
    }

    refreshButton.addEventListener("click", loadMonitoring);
    loadMonitoring();
    setInterval(loadMonitoring, 5000);
  </script>
</body>
</html>
"""


@dataclass
class ServiceState:
    started_at: datetime
    prediction_requests: int = 0
    prediction_errors: int = 0
    total_prediction_latency_seconds: float = 0.0


state = ServiceState(started_at=datetime.now(UTC))
app = FastAPI(
    title="Mushroom Edibility Classifier",
    description="FastAPI service for binary mushroom image classification.",
    version="0.1.0",
)


def get_model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()


def get_metrics_path() -> Path:
    return Path(os.getenv("METRICS_PATH", DEFAULT_METRICS_PATH)).resolve()


def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "resnet18")


def get_device() -> torch.device:
    requested_device = os.getenv("DEVICE", "auto").lower()

    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint() -> dict[str, Any]:
    model_path = get_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return torch.load(model_path, map_location=get_device())


def load_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_checkpoint()
    model_name = checkpoint.get("model_name", get_model_name())
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    image_size = int(checkpoint.get("image_size", 224))

    if class_names != CLASS_NAMES:
        raise ValueError(f"Unexpected model classes: {class_names}. Expected: {CLASS_NAMES}")

    model = build_model(
        model_name=model_name,
        pretrained=False,
        freeze_backbone=False,
        num_classes=len(CLASS_NAMES),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(get_device())
    model.eval()

    metadata = {
        "model_name": model_name,
        "model_path": str(get_model_path()),
        "class_names": class_names,
        "image_size": image_size,
        "device": str(get_device()),
    }
    return model, metadata


def load_model_quality_metrics() -> dict[str, Any]:
    metrics_path = get_metrics_path()
    base_response: dict[str, Any] = {
        "available": False,
        "source_path": str(metrics_path),
        "split": "test",
        "positive_class": "non_edible",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
        "roc-auc": None,
        "load_error": None,
    }

    if not metrics_path.exists():
        return {
            **base_response,
            "load_error": f"Metrics file not found: {metrics_path}",
        }

    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base_response,
            "load_error": f"Could not read metrics file: {exc}",
        }

    test_metrics = payload.get("test_metrics", {})
    return {
        **base_response,
        "available": True,
        "positive_class": payload.get("positive_class", "non_edible"),
        "accuracy": test_metrics.get("accuracy"),
        "precision": test_metrics.get("precision_non_edible", test_metrics.get("precision")),
        "recall": test_metrics.get("recall_non_edible", test_metrics.get("recall")),
        "f1": test_metrics.get("f1_non_edible", test_metrics.get("f1")),
        "roc_auc": test_metrics.get("roc_auc"),
        "roc-auc": test_metrics.get("roc_auc"),
    }


try:
    MODEL, MODEL_METADATA = load_model()
    MODEL_LOAD_ERROR = None
except Exception as exc:  # noqa: BLE001 - service must start even if model is absent.
    MODEL = None
    MODEL_METADATA = {
        "model_name": get_model_name(),
        "model_path": str(get_model_path()),
        "class_names": CLASS_NAMES,
        "image_size": 224,
        "device": str(get_device()),
    }
    MODEL_LOAD_ERROR = str(exc)


def read_uploaded_image(file: UploadFile) -> Image.Image:
    try:
        image = Image.open(file.file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    return image


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "mushroom-edibility-classifier",
        "predict_endpoint": "/predict",
        "monitoring_endpoint": "/monitoring",
        "ui_endpoint": "/ui",
    }


@app.get("/ui", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    started = perf_counter()
    state.prediction_requests += 1

    if MODEL is None:
        state.prediction_errors += 1
        raise HTTPException(
            status_code=503,
            detail=f"Model is not available: {MODEL_LOAD_ERROR}",
        )

    try:
        image = read_uploaded_image(file)
        transform = build_transforms(MODEL_METADATA["image_size"])["test"]
        tensor = transform(image).unsqueeze(0).to(get_device())

        with torch.no_grad():
            logits = MODEL(tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        predicted_index = int(torch.tensor(probabilities).argmax().item())
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index])
        latency = perf_counter() - started
        state.total_prediction_latency_seconds += latency

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                class_name: float(probability)
                for class_name, probability in zip(CLASS_NAMES, probabilities, strict=False)
            },
            "latency_seconds": latency,
            "model": MODEL_METADATA,
        }
    except HTTPException:
        state.prediction_errors += 1
        raise
    except Exception as exc:  # noqa: BLE001 - convert inference failures to API errors.
        state.prediction_errors += 1
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/monitoring")
def monitoring() -> dict[str, Any]:
    uptime_seconds = (datetime.now(UTC) - state.started_at).total_seconds()
    successful_predictions = max(0, state.prediction_requests - state.prediction_errors)
    average_latency = (
        state.total_prediction_latency_seconds / successful_predictions
        if successful_predictions > 0
        else 0.0
    )

    process = psutil.Process()
    memory = process.memory_info()

    return {
        "service": {
            "status": "ok" if MODEL is not None else "degraded",
            "started_at": state.started_at.isoformat(),
            "uptime_seconds": uptime_seconds,
        },
        "model": {
            "loaded": MODEL is not None,
            "load_error": MODEL_LOAD_ERROR,
            **MODEL_METADATA,
        },
        "prediction_metrics": {
            "requests_total": state.prediction_requests,
            "errors_total": state.prediction_errors,
            "successful_predictions_total": successful_predictions,
            "average_latency_seconds": average_latency,
        },
        "infrastructure": {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "process_rss_mb": memory.rss / (1024 * 1024),
            "process_vms_mb": memory.vms / (1024 * 1024),
            "cuda_available": torch.cuda.is_available(),
            "device": str(get_device()),
        },
        "metrics": load_model_quality_metrics(),
    }
