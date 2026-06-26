from __future__ import annotations

from typing import Any

from models.base import DetectionSegModel

# Published headline metrics for the detectors in the benchmark. COCO mAP unless noted.
PUBLISHED_MAP: dict[str, dict] = {
    "YOLO11n": {"map": 39.5, "note": "COCO val mAP (nano)"},
    "RF-DETR-nano": {"map": 48.4, "note": "COCO val mAP (nano)"},
    "YOLO-World": {"map": 35.4, "note": "LVIS zero-shot AP (small)"},
}


def measure_latency(model: DetectionSegModel, image: Any, **kwargs: Any) -> float:
    return model.predict(image, **kwargs).latency_ms


def build_benchmark_rows(measured: "dict[str, float]") -> "list[dict]":
    rows: list[dict] = []
    for name, latency in measured.items():
        if name in PUBLISHED_MAP:
            rows.append(
                {
                    "model": name,
                    "map": PUBLISHED_MAP[name]["map"],
                    "latency_ms": latency,
                    "note": PUBLISHED_MAP[name]["note"],
                }
            )
    return rows
