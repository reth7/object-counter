FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# torch AND torchvision from the same CPU index — mixing indexes causes
# the "torchvision::nms does not exist" version-mismatch error
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir timm safetensors

# Plan C: weights downloaded manually once, baked into the image.
# Put the file at weights/dinov2_vits14.safetensors before building.
COPY weights/ /srv/weights/
COPY app/ ./app/

# Sanity check at build time: model must construct and load local weights
RUN python -c "import os; os.environ['DINO_WEIGHTS']='/srv/weights/dinov2_vits14.safetensors'; \
import sys; sys.path.insert(0,'/srv'); \
from app.engine import _load_dino; m=_load_dino(); \
assert m is not None, 'DINOv2 failed to load - check weights file'; print('DINOv2 OK')"

ENV OMP_NUM_THREADS=1 OPENCV_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_THREADS=1 \
    DINO_WEIGHTS=/srv/weights/dinov2_vits14.safetensors
EXPOSE 8090
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
