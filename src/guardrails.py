from __future__ import annotations

from typing import Any

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
REQUIRED_KEYS = {"image_quality", "predicted_class", "confidence", "visual_evidence", "justification", "limitations", "warning"}
# Drapeaux de qualité « basse » : toy backend en anglais (limited/poor),
# MedGemma en français (mauvaise). Sans le terme FR, le repli sur `uncertain`
# ne se déclenchait jamais pour MedGemma.
LOW_QUALITY_FLAGS = {"limited", "poor", "mauvaise"}
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."


def validate_prediction(pred: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_KEYS - set(pred)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if pred.get("predicted_class") not in ALLOWED_CLASSES:
        errors.append("invalid predicted_class")
    try:
        conf = float(pred.get("confidence", -1))
        if not 0 <= conf <= 1:
            errors.append("confidence outside [0,1]")
    except Exception:
        errors.append("confidence is not numeric")
    if not pred.get("warning"):
        errors.append("warning missing")
    return not errors, errors


def apply_safety_guardrails(pred: dict[str, Any]) -> dict[str, Any]:
    valid, errors = validate_prediction(pred)
    if not valid:
        pred["predicted_class"] = "uncertain"
        pred["confidence"] = min(float(pred.get("confidence", 0.0) or 0.0), 0.5)
        # `limitations` est une liste côté toy backend mais une chaîne côté MedGemma
        # (cf. schéma des prompts). On préserve le type d'origine pour rester
        # cohérent avec chaque backend, sans présumer d'un `.append`.
        note = "guardrail triggered: invalid output schema"
        lims = pred.get("limitations")
        if isinstance(lims, list):
            lims.append(note)
        elif lims:
            pred["limitations"] = f"{lims} | {note}"
        else:
            pred["limitations"] = note
    if pred.get("image_quality") in LOW_QUALITY_FLAGS and float(pred.get("confidence", 0)) < 0.6:
        pred["predicted_class"] = "uncertain"
    pred["warning"] = WARNING_TEXT
    pred["guardrail_errors"] = errors
    return pred
