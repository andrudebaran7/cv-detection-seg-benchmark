from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from models.base import Prediction

_BOX_COLOR = (255, 0, 0)
_MASK_COLOR = np.array([0, 200, 255], dtype=np.uint8)


def draw_prediction(image: "Image.Image", prediction: Prediction) -> "Image.Image":
    out = image.convert("RGB").copy()

    if prediction.masks:
        arr = np.array(out)
        for mask in prediction.masks:
            m = np.array(mask)
            if m.shape[:2] != arr.shape[:2]:
                m = np.array(
                    Image.fromarray((np.array(mask) * 255).astype("uint8"))
                    .resize((arr.shape[1], arr.shape[0]))
                ) > 127
            sel = m.astype(bool)
            arr[sel] = (0.5 * arr[sel] + 0.5 * _MASK_COLOR).astype(np.uint8)
        out = Image.fromarray(arr)

    draw = ImageDraw.Draw(out)
    for box, label, score in zip(prediction.boxes, prediction.labels, prediction.scores):
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=_BOX_COLOR, width=2)
        draw.text((box.x1, max(0, box.y1 - 10)), f"{label} {score:.2f}", fill=_BOX_COLOR)

    return out
