"""
Object Counting API for Vrishabh Industries ERP
================================================
Exemplar-based counting: user uploads (1) a crop of the object (template)
and (2) the scene image. Returns count + annotated image.

Security: API key + HMAC-SHA256 request signing, size/type validation,
rate limiting. Retention: images auto-deleted after 7 days.
"""

import hashlib
import hmac
import os
import time
import uuid
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from apscheduler.schedulers.background import BackgroundScheduler

# ---------------------------------------------------------------- config
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", "/data/uploads"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "7"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
API_KEYS = {}  # key_id -> secret, loaded from env: API_KEYS="erp1:secret1,erp2:secret2"
for pair in os.environ.get("API_KEYS", "").split(","):
    if ":" in pair:
        kid, secret = pair.split(":", 1)
        API_KEYS[kid.strip()] = secret.strip()

DEV_MODE = os.environ.get("DEV_MODE", "0") == "1"  # skips auth — local testing only!
SIGNATURE_TOLERANCE_SEC = 300  # reject requests older than 5 min (replay protection)
ALLOWED_MAGIC = {
    b"\xff\xd8\xff": "jpg",
    b"\x89PNG\r\n\x1a\n": "png",
}

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Vrishabh Object Counter", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------- rate limit (simple in-memory token bucket per key)
_buckets: dict = {}
RATE_LIMIT = 30        # requests
RATE_WINDOW = 60       # per 60 seconds


def check_rate(key_id: str):
    now = time.time()
    hits = _buckets.setdefault(key_id, [])
    hits[:] = [t for t in hits if now - t < RATE_WINDOW]
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded")
    hits.append(now)


# ---------------------------------------------------------------- auth
def verify_request(key_id: str, timestamp: str, signature: str, body_digest: str):
    if DEV_MODE:
        return
    if key_id not in API_KEYS:
        raise HTTPException(401, "Unknown API key")
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        raise HTTPException(401, "Bad timestamp")
    if abs(time.time() - ts) > SIGNATURE_TOLERANCE_SEC:
        raise HTTPException(401, "Request expired")
    expected = hmac.new(
        API_KEYS[key_id].encode(),
        f"{key_id}.{timestamp}.{body_digest}".encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, signature or ""):
        raise HTTPException(401, "Invalid signature")
    check_rate(key_id)


def validate_image_bytes(data: bytes, label: str) -> np.ndarray:
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{label} exceeds {MAX_UPLOAD_MB} MB")
    if not any(data.startswith(m) for m in ALLOWED_MAGIC):
        raise HTTPException(415, f"{label} must be JPEG or PNG")
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(422, f"{label} could not be decoded")
    return arr


from app.engine import count_objects


# ---------------------------------------------------------------- endpoints
WEB_DIR = Path(__file__).parent / "web"


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "retention_days": RETENTION_DAYS}


@app.post("/v1/count")
async def count(
    request: Request,
    template: UploadFile = File(...),
    scene: UploadFile = File(...),
    x_api_key: str = Header(None),
    x_timestamp: str = Header(None),
    x_signature: str = Header(None),
    allow_rotation: bool = False,
    inline: bool = False,
):
    t_bytes = await template.read()
    s_bytes = await scene.read()
    body_digest = hashlib.sha256(t_bytes + s_bytes).hexdigest()
    verify_request(x_api_key, x_timestamp, x_signature, body_digest)

    tmpl_img = validate_image_bytes(t_bytes, "template")
    scene_img = validate_image_bytes(s_bytes, "scene")

    rotations = (0, 90, 180, 270) if allow_rotation else (0,)
    t0 = time.time()
    n, boxes, annotated, sf, conf = count_objects(tmpl_img, scene_img, rotations=rotations)
    elapsed = round(time.time() - t0, 2)

    # persist with date-stamped folder for retention sweeps
    job_id = uuid.uuid4().hex[:12]
    day_dir = STORAGE_DIR / datetime.utcnow().strftime("%Y-%m-%d")
    day_dir.mkdir(exist_ok=True)
    (day_dir / f"{job_id}_template.jpg").write_bytes(t_bytes)
    (day_dir / f"{job_id}_scene.jpg").write_bytes(s_bytes)
    cv2.imwrite(str(day_dir / f"{job_id}_annotated.jpg"), annotated)

    payload = {
        "job_id": job_id,
        "count": n,
        "confidence": conf,
        "boxes": boxes,
        "scene_scale_applied": sf,
        "processing_seconds": elapsed,
        "annotated_url": f"/v1/result/{job_id}/annotated",
        "expires": (datetime.utcnow() + timedelta(days=RETENTION_DAYS)).isoformat() + "Z",
    }
    if inline:
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            import base64
            payload["annotated_b64"] = base64.b64encode(buf.tobytes()).decode()
    return JSONResponse(payload)



@app.get("/v1/result/{job_id}/annotated")
def get_annotated(job_id: str, x_api_key: str = Header(None),
                  x_timestamp: str = Header(None), x_signature: str = Header(None)):
    if not job_id.isalnum() or len(job_id) > 16:
        raise HTTPException(400, "Bad job id")
    verify_request(x_api_key, x_timestamp, x_signature, hashlib.sha256(job_id.encode()).hexdigest())
    for day_dir in STORAGE_DIR.iterdir():
        f = day_dir / f"{job_id}_annotated.jpg"
        if f.exists():
            return FileResponse(f, media_type="image/jpeg")
    raise HTTPException(404, "Result not found or expired")


# ---------------------------------------------------------------- retention sweeper
def purge_old():
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    for day_dir in STORAGE_DIR.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            d = datetime.strptime(day_dir.name, "%Y-%m-%d")
        except ValueError:
            continue
        if d < cutoff:
            shutil.rmtree(day_dir, ignore_errors=True)


scheduler = BackgroundScheduler()
scheduler.add_job(purge_old, "cron", hour=2, minute=30)  # daily 02:30 UTC
scheduler.start()
purge_old()  # sweep on startup too
