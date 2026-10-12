# ERP Integration — Client Examples

Every request must carry three headers:

| Header        | Value                                                            |
|---------------|------------------------------------------------------------------|
| `X-Api-Key`   | your key id, e.g. `erp_main`                                     |
| `X-Timestamp` | current unix time (seconds)                                      |
| `X-Signature` | `HMAC_SHA256(secret, "{key_id}.{timestamp}.{sha256(template_bytes + scene_bytes)}")` hex |

Requests older than 5 minutes or with a wrong signature are rejected (replay protection).

## Python client (e.g. from a cron job or ERP backend)

```python
import hashlib, hmac, time, requests

KEY_ID = "erp_main"
SECRET = "LONG_RANDOM_SECRET"
URL = "https://console.vrishabhindustries.in/counter-api/v1/count"

t = open("cone_template.jpg", "rb").read()
s = open("carton_photo.jpg", "rb").read()

ts = str(int(time.time()))
digest = hashlib.sha256(t + s).hexdigest()
sig = hmac.new(SECRET.encode(), f"{KEY_ID}.{ts}.{digest}".encode(), hashlib.sha256).hexdigest()

r = requests.post(
    URL,
    headers={"X-Api-Key": KEY_ID, "X-Timestamp": ts, "X-Signature": sig},
    files={"template": ("t.jpg", t, "image/jpeg"), "scene": ("s.jpg", s, "image/jpeg")},
    params={"allow_rotation": "false"},
    timeout=60,
)
print(r.json())   # {"job_id": "...", "count": 48,
  "confidence": 0.91, ...}
```

## PHP client (if your ERP backend is PHP)

```php
$keyId  = 'erp_main';
$secret = 'LONG_RANDOM_SECRET';
$t = file_get_contents('cone_template.jpg');
$s = file_get_contents('carton_photo.jpg');

$ts     = (string) time();
$digest = hash('sha256', $t . $s);
$sig    = hash_hmac('sha256', "$keyId.$ts.$digest", $secret);

$ch = curl_init('https://console.vrishabhindustries.in/counter-api/v1/count');
curl_setopt_array($ch, [
  CURLOPT_POST => true,
  CURLOPT_RETURNTRANSFER => true,
  CURLOPT_HTTPHEADER => [
    "X-Api-Key: $keyId", "X-Timestamp: $ts", "X-Signature: $sig",
  ],
  CURLOPT_POSTFIELDS => [
    'template' => new CURLStringFile($t, 't.jpg', 'image/jpeg'),
    'scene'    => new CURLStringFile($s, 's.jpg', 'image/jpeg'),
  ],
]);
echo curl_exec($ch);
```

## Response

```json
{
  "job_id": "a1b2c3d4e5f6",
  "count": 48,
  "confidence": 0.91,
  "boxes": [[x1,y1,x2,y2], ...],
  "processing_seconds": 1.8,
  "annotated_url": "/v1/result/a1b2c3d4e5f6/annotated",
  "expires": "2026-06-19T10:00:00Z"
}
```

Show `annotated_url` to the user in the ERP so the operator can visually verify
the boxes before accepting the count into a lot/dispatch record.

## Tuning

- There is no threshold parameter — the engine picks the accept/reject cut automatically and
  returns a `confidence` score (0.5–0.99). In the ERP, flag counts with confidence < 0.75
  for mandatory operator review.
- `allow_rotation=true`: enables 90°/180°/270° matching (4x slower) — use only if objects appear rotated.
- Template quality matters most: crop ONE object tightly, from the same photo or same camera angle/lighting as the scene.
- For cone counting, photograph cartons top-down so all cone tips look identical.
