"""
Counting engine: propose -> verify -> auto-threshold.

1. PROPOSE  - bootstrap template matching at low threshold (high recall).
2. VERIFY   - embed every candidate with DINOv2 ViT-S/14; cosine similarity
              to an exemplar prototype separates objects from background.
              Falls back to HOG+color features if torch/DINOv2 unavailable.
3. DECIDE   - accept/reject cut chosen automatically from the similarity
              distribution (largest-gap). No user-facing tuning parameter.
"""
import os
import cv2
import numpy as np

# ------------------------------------------------------------------ verifier
_DINO = {"model": None, "tried": False}


def _load_dino():
    if _DINO["tried"]:
        return _DINO["model"]
    _DINO["tried"] = True
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
        import timm
        local = os.environ.get("DINO_WEIGHTS", "/srv/weights/dinov2_vits14.safetensors")
        if os.path.exists(local):
            # Plan C: weights baked into the image, zero network needed
            from safetensors.torch import load_file
            from timm.layers import resample_abs_pos_embed
            model = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m",
                pretrained=False, num_classes=0, img_size=126)
            sd = load_file(local)
            sd.pop("head.weight", None); sd.pop("head.bias", None)
            if "pos_embed" in sd and sd["pos_embed"].shape != model.pos_embed.shape:
                grid = model.patch_embed.grid_size
                sd["pos_embed"] = resample_abs_pos_embed(
                    sd["pos_embed"], new_size=list(grid), num_prefix_tokens=1)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[engine] loaded local DINOv2 weights "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            # network paths: HF via timm, then torch.hub/GitHub
            try:
                model = timm.create_model(
                    "vit_small_patch14_dinov2.lvd142m",
                    pretrained=True, num_classes=0, img_size=126)
            except Exception:
                model = torch.hub.load("facebookresearch/dinov2",
                                       "dinov2_vits14", verbose=False)
        model.eval()
        _DINO["model"] = model
    except Exception as e:  # noqa: BLE001
        print(f"[engine] DINOv2 unavailable ({e}); using HOG fallback verifier")
        _DINO["model"] = None
    return _DINO["model"]


def _dino_embed(patches_bgr):
    """patches_bgr: list of HxWx3 uint8 -> (N, 384) L2-normalized features."""
    import torch
    model = _DINO["model"]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    batch = []
    for p in patches_bgr:
        rgb = cv2.cvtColor(cv2.resize(p, (126, 126)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - mean) / std
        batch.append(x.transpose(2, 0, 1))
    t = torch.from_numpy(np.stack(batch))
    with torch.inference_mode():
        f = model(t).numpy()
    f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    return f


_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def _hog_embed(patches_bgr):
    feats = []
    for p in patches_bgr:
        g = cv2.resize(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), (64, 64))
        h = _HOG.compute(g).flatten()
        hsv = cv2.cvtColor(cv2.resize(p, (64, 64)), cv2.COLOR_BGR2HSV)
        ch = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()
        v = np.concatenate([h / (np.linalg.norm(h) + 1e-9),
                            0.5 * ch / (np.linalg.norm(ch) + 1e-9)])
        feats.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(feats)


def _embed(patches_bgr):
    if _load_dino() is not None:
        return _dino_embed(patches_bgr)
    return _hog_embed(patches_bgr)


# ------------------------------------------------------------------ proposals
def _clahe_gray(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    return cv2.createCLAHE(2.0, (8, 8)).apply(g)


def nms(boxes, scores, iou_thresh=0.35):
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, np.float32)
    scores = np.asarray(scores, np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def _match_bank(sg, bank, scales, rotations=(0,), low=0.42, cap=600):
    boxes, scores = [], []
    for t in bank:
        h0, w0 = t.shape
        for rot in rotations:
            if rot:
                M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), rot, 1.0)
                t_r = cv2.warpAffine(t, M, (w0, h0))
            else:
                t_r = t
            for s in scales:
                tw, th = int(w0 * s), int(h0 * s)
                if tw < 12 or th < 12 or tw >= sg.shape[1] or th >= sg.shape[0]:
                    continue
                res = cv2.matchTemplate(sg, cv2.resize(t_r, (tw, th)),
                                        cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= low)
                if len(xs) > cap:  # keep top responses only
                    idx = np.argsort(res[ys, xs])[::-1][:cap]
                    ys, xs = ys[idx], xs[idx]
                for x, y in zip(xs, ys):
                    boxes.append([x, y, x + tw, y + th])
                    scores.append(float(res[y, x]))
    return boxes, scores


def _propose(template, scene, rotations=(0,)):
    sg, tg = _clahe_gray(scene), _clahe_gray(template)
    # seed pass with user's exemplar
    b, s = _match_bank(sg, [tg], [0.85, 0.92, 1.0, 1.08, 1.15], rotations)
    if not b:
        return np.empty((0, 4), int), np.empty((0, 4), int)
    b, s = np.array(b, np.float32), np.array(s, np.float32)
    k = nms(b, s)
    kb, ks = b[k], s[k]
    seeds = kb[ks >= max(0.50, ks.max() * 0.60)].astype(int)
    # bootstrap bank from found instances, rematch at low threshold
    bank = [tg]
    for (x1, y1, x2, y2) in seeds[:6]:
        patch = sg[y1:y2, x1:x2]
        if patch.size:
            bank.append(patch)
    b2, s2 = _match_bank(sg, bank, [0.9, 1.0, 1.1], rotations)
    b2, s2 = np.array(b2, np.float32), np.array(s2, np.float32)
    k2 = nms(b2, s2)
    return b2[k2].astype(int), seeds


# ------------------------------------------------------------------ decision
def _auto_threshold(sims):
    """Largest-gap cut on sorted similarities; floors keep degenerate
    cases sane (all-true or all-noise candidate sets)."""
    ss = np.sort(sims)[::-1]
    if len(ss) <= 1:
        return 0.5
    gaps = ss[:-1] - ss[1:]
    valid = [(g, i) for i, g in enumerate(gaps) if ss[i + 1] < 0.92 * ss[0]]
    if valid:
        g, i = max(valid)
        if g > 0.06:
            return float((ss[i] + ss[i + 1]) / 2)
    return 0.55  # no clear gap -> conservative absolute floor


def _crop(scene, box, pad=0.12):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad), int(h * pad)
    H, W = scene.shape[:2]
    return scene[max(0, y1 - py):min(H, y2 + py),
                 max(0, x1 - px):min(W, x2 + px)]


def count_objects(template, scene, rotations=(0,)):
    """Returns (count, boxes, annotated_image, scale_factor, confidence)."""
    max_dim = 1600
    scale_factor = 1.0
    h, w = scene.shape[:2]
    if max(h, w) > max_dim:
        scale_factor = max_dim / max(h, w)
        scene = cv2.resize(scene, None, fx=scale_factor, fy=scale_factor)
        template = cv2.resize(template, None, fx=scale_factor, fy=scale_factor)

    cand, seeds = _propose(template, scene, rotations)
    if len(cand) == 0:
        return 0, [], scene, scale_factor, 1.0

    # prototype = exemplar + up to 5 confident seed instances
    proto_patches = [template] + [_crop(scene, b) for b in seeds[:5]]
    proto_patches = [p for p in proto_patches if p.size]
    cand_patches = [_crop(scene, b) for b in cand]
    keep_idx = [i for i, p in enumerate(cand_patches) if p.size]
    cand = cand[keep_idx]
    cand_patches = [cand_patches[i] for i in keep_idx]

    feats = _embed(proto_patches + cand_patches)
    proto = feats[:len(proto_patches)].mean(axis=0)
    proto /= (np.linalg.norm(proto) + 1e-9)
    sims = feats[len(proto_patches):] @ proto

    thr = _auto_threshold(sims)
    accept = sims >= thr
    final = cand[accept]
    acc_sims = sims[accept]

    # confidence: how cleanly accepted/rejected separate
    rej = sims[~accept]
    margin = float(acc_sims.min() - rej.max()) if (len(acc_sims) and len(rej)) else 0.3
    confidence = float(np.clip(0.5 + margin * 2.0, 0.5, 0.99))

    annotated = scene.copy()
    for i, (x1, y1, x2, y2) in enumerate(final, 1):
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, str(i), (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return len(final), final.tolist(), annotated, scale_factor, confidence
