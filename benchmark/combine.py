from __future__ import annotations

import csv


def load_combined(*paths) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                r["resolution"] = int(r["resolution"])
                r["value"] = float(r["value"])
                r["n_iters"] = int(r["n_iters"])
                rows.append(r)
    return rows


def value_for(rows, *, device, model, experiment, metric, resolution):
    for r in rows:
        if (r["device"] == device and r["model"] == model and r["experiment"] == experiment
                and r["metric"] == metric and r["resolution"] == resolution):
            return r["value"]
    return None


def models_for_task(rows, task) -> list[str]:
    return sorted({r["model"] for r in rows if r["task"] == task})
