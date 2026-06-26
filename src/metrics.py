from __future__ import annotations

from collections import Counter
from typing import Iterable

CLASSES = ["normal", "suspected_opacity", "uncertain"]

# Seuils de qualité du projet (operator, threshold)
TARGETS: dict[str, tuple[str, float]] = {
    "accuracy":        (">=", 0.70),
    "macro_f1":        (">=", 0.68),
    "json_valid_rate": (">=", 0.95),
    "avg_latency_ms":  ("<",  10_000),
}


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append(f1)
    return sum(scores) / len(scores)


def confusion_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        counts[f"{t}__{p}"] += 1
    return dict(counts)


def summarize_metrics(rows: list[dict]) -> dict:
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    json_valid = [r.get("json_valid", True) for r in rows]
    warnings = [bool(r.get("warning")) for r in rows]

    metrics: dict = {
        "n": len(rows),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "json_valid_rate": round(sum(json_valid) / len(json_valid), 4) if rows else 0,
        "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in rows) / len(rows)) if rows else 0,
        "warning_rate": round(sum(warnings) / len(warnings), 4) if rows else 0,
        "uncertain_rate": round(sum(p == "uncertain" for p in y_pred) / len(y_pred), 4) if rows else 0,
    }

    def _pass(op: str, value: float, threshold: float) -> bool:
        return value >= threshold if op == ">=" else value < threshold

    targets_block = {}
    for key, (op, threshold) in TARGETS.items():
        value = metrics.get(key, 0)
        targets_block[key] = {"threshold": threshold, "operator": op, "pass": _pass(op, value, threshold)}

    metrics["targets"] = targets_block
    metrics["all_targets_met"] = all(v["pass"] for v in targets_block.values())
    return metrics
