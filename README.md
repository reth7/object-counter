# Object Counter

Backend microservice for the **Vrishabh Industries ERP**. It counts identical objects
in a photo of a factory floor, a carton, or a pallet — cones, caps, fasteners, boxes,
anything visually repetitive — so operators don't have to count them by hand.

Given a cropped photo of one object (the "template") and a wider photo of the pile or
batch (the "scene"), the API returns the count, bounding boxes, a confidence score, and
an annotated image the operator can visually verify before the count is accepted into
a lot or dispatch record.

## Why this exists

Manually counting identical small parts on the factory floor is slow and error-prone.
This service automates that step and plugs into the ERP as a simple HTTP API, so a
count that used to take a person several minutes takes a couple of seconds.

## How it works

1. **Propose** — template matching against the scene at multiple scales/rotations,
   bootstrapped from the user's exemplar (high recall, including false positives).
2. **Verify** — every candidate patch is embedded with a DINOv2 ViT-S/14 vision model
   and compared (cosine similarity) against a prototype built from the exemplar and a
   few confident seed matches. Falls back to HOG + color histogram features if PyTorch
   or the DINOv2 weights aren't available.
3. **Decide** — the accept/reject similarity cutoff is chosen automatically from the
   gap in the similarity distribution. There is no threshold for the operator to tune;
   instead the API returns a `confidence` score so the ERP can flag low-confidence
   counts for manual review.

This keeps the model lightweight enough to run on a single CPU core, which is the point:
it's meant to run on **modest, on-prem hardware** next to other factory systems, not on
a GPU server.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `POST /v1/count` | Upload `template` + `scene` images, get back count + boxes + confidence |
| `GET /v1/result/{job_id}/annotated` | Fetch the annotated scene image for a prior count |

Every request to `/v1/count` and `/v1/result/...` must carry `X-Api-Key`, `X-Timestamp`,
and an HMAC-SHA256 `X-Signature` (replay-protected, 5-minute window). See
[ERP_INTEGRATION.md](ERP_INTEGRATION.md) for the signing scheme and working Python/PHP
client examples.

Uploaded images and results are stored under `STORAGE_DIR` and purged automatically
after `RETENTION_DAYS` (default 7).

## Running locally

```bash
docker compose -f docker-compose.dev.yml up --build
```

This starts the service with auth disabled (`DEV_MODE=1`) on `http://localhost:8090`,
with a minimal web UI at `/` for trying it out from a browser or phone on the LAN.

## Testing the app with Docker

The dev compose file (`docker-compose.dev.yml`) is the easiest way to try the whole
flow end-to-end — upload a template, upload a scene, get a count back — without
setting up API keys or a reverse proxy.

1. **Make sure Docker Desktop is running**, then from the project root build and start
   the dev container:

   ```bash
   docker compose -f docker-compose.dev.yml up --build
   ```

   First build takes a few minutes (installs PyTorch/timm and bakes in the DINOv2
   weights). Leave this running in the terminal — it logs each request.

2. **Confirm it's up** by checking the health endpoint:

   ```bash
   curl http://localhost:8090/health
   # {"status":"ok","retention_days":7}
   ```

3. **Find the local IP to use.** `docker-compose.dev.yml` binds the container to
   `0.0.0.0:8090`, so it's reachable from any device on the same Wi-Fi/LAN — useful for
   testing from a phone camera against a factory-floor photo. Find the machine's LAN
   IPv4 address:

   - **Windows (PowerShell):** `ipconfig` → look for "IPv4 Address" under your active
     adapter (usually `192.168.x.x`).
   - **macOS:** `ipconfig getifaddr en0`
   - **Linux:** `hostname -I`

   Then open `http://<that-ip>:8090` — e.g. `http://192.168.1.42:8090` — on your phone
   or another computer on the same network. Use `http://localhost:8090` if you're
   testing on the same machine that's running Docker.

4. **Use the test console** at that URL: take/choose a photo, draw a tight box around
   one object to use as the template, submit, and the annotated result with the count
   comes back in a few seconds.

   You can also hit the API directly, e.g. with `curl` or the Python/PHP snippets in
   [ERP_INTEGRATION.md](ERP_INTEGRATION.md) (set `DEV_MODE=1` as above to skip the
   HMAC signing while testing).

5. **Stop the container** with `Ctrl+C`, or `docker compose -f docker-compose.dev.yml down`.

> Only use `docker-compose.dev.yml` for testing on a trusted LAN. It has no
> authentication and no CPU/RAM limits. Use the plain `docker-compose.yml` (with API
> keys and a reverse proxy) for anything production-facing.

## Deploying

```bash
docker compose up --build -d
```

Production config (`docker-compose.yml`) pins the container to a single vCPU and
~2 GB RAM, binds it to `localhost` only, and expects a reverse proxy (see
[nginx/counter-api.conf](nginx/counter-api.conf)) in front of it for TLS and the
`/counter-api/` path prefix used by the ERP. Set `COUNTER_API_KEYS` in a `.env` file
(`key_id:secret` pairs, comma-separated) before starting.

### DINOv2 weights

The verifier needs `weights/dinov2_vits14.safetensors` present before the Docker image
is built — see [weights/README_WEIGHTS.txt](weights/README_WEIGHTS.txt). Without it,
the service still works but falls back to the weaker HOG-based verifier.

## Resource footprint

Designed to run on light hardware: 1 CPU core, ~2 GB RAM, no GPU required. Image
resolution is capped internally (1600px max dimension) to keep per-request latency and
memory use predictable regardless of the photo size an operator uploads.

## Screenshots

Drop screenshots of the test console / interface and example counting results into
`docs/screenshots/` and reference them below (markdown preview will render them once
the files exist):

<!-- ![Test console](docs/screenshots/test-console.png) -->
<!-- ![Counting result](docs/screenshots/counting-result.png) -->
<!-- ![ERP integration view](docs/screenshots/erp-integration.png) -->

## Project layout

```
app/
  main.py      FastAPI app: auth, rate limiting, upload handling, retention sweeper
  engine.py    Counting engine: propose -> verify -> auto-threshold
  web/         Minimal browser UI for manual testing
weights/       DINOv2 model weights (not committed; see README_WEIGHTS.txt)
nginx/         Reverse proxy config snippet for the ERP server
docker-compose.yml       Production deployment (pinned CPU/RAM, auth required)
docker-compose.dev.yml   Local/LAN testing (no auth, no resource limits)
ERP_INTEGRATION.md       Request signing scheme + client code samples
```
