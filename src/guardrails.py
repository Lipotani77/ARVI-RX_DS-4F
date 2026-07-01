from __future__ import annotations

from typing import Any

from .safety_classifier import classify_image

ALLOWED_CLASSES = {"normal", "suspected_opacity", "uncertain"}
REQUIRED_KEYS = {"image_quality", "predicted_class", "confidence", "visual_evidence", "justification", "limitations", "warning"}
# Drapeaux de qualité « basse » : toy backend en anglais (limited/poor),
# MedGemma en français (mauvaise). Sans le terme FR, le repli sur `uncertain`
# ne se déclenchait jamais pour MedGemma.
LOW_QUALITY_FLAGS = {"limited", "poor", "mauvaise"}
WARNING_TEXT = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Seuil du garde-fou (CNN)
SAFETY_CONFIDENCE_FLOOR = 0.55

# Seuil de confiance de MedGemma
MAIN_CONFIDENCE_FLOOR = 0.6


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


def _coerce_confidence(pred: dict[str, Any]) -> float:
    try:
        return float(pred.get("confidence", 0.0) or 0.0)
    except Exception:
        return 0.0


def apply_safety_guardrails(pred: dict[str, Any], image_path: str | None = None) -> dict[str, Any]:

    # valid est false si des clés manquent 
    valid, errors = validate_prediction(pred)
    safety_pred: dict[str, Any] | None = None

    if image_path is not None:
        safety_pred = classify_image(image_path)
        pred["safety_predicted_class"] = safety_pred["predicted_class"]
        pred["safety_confidence"] = safety_pred["confidence"]
        pred["safety_probabilities"] = safety_pred["probabilities"]

    main_conf = _coerce_confidence(pred)

    if not valid:
        if safety_pred and safety_pred["confidence"] >= SAFETY_CONFIDENCE_FLOOR:
            pred["predicted_class"] = safety_pred["predicted_class"]
            pred["confidence"] = safety_pred["confidence"]
        else:
            pred["predicted_class"] = "uncertain"
            pred["confidence"] = min(main_conf, 0.5)

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

    # Si la qualité de l'image est basse ET que la confiance du modèle principal est
    # inférieure au seuil, on repli sur le modèle de garde-fou 
    if pred.get("image_quality") in LOW_QUALITY_FLAGS and _coerce_confidence(pred) < MAIN_CONFIDENCE_FLOOR:
        if safety_pred and safety_pred["confidence"] >= SAFETY_CONFIDENCE_FLOOR:
            pred["predicted_class"] = safety_pred["predicted_class"]
            pred["confidence"] = safety_pred["confidence"]
        else:
            pred["predicted_class"] = "uncertain"

    # Si la confiance du modèle principal est inférieure au seuil, on repli sur le modèle de garde-fou si sa confiance est suffisante, sinon on repli sur `uncertain
    if safety_pred and _coerce_confidence(pred) < MAIN_CONFIDENCE_FLOOR:
        if safety_pred["confidence"] >= SAFETY_CONFIDENCE_FLOOR:
            pred["predicted_class"] = safety_pred["predicted_class"]
            pred["confidence"] = safety_pred["confidence"]
        else:
            pred["predicted_class"] = "uncertain"
            pred["confidence"] = min(_coerce_confidence(pred), safety_pred["confidence"])

    if _coerce_confidence(pred) < SAFETY_CONFIDENCE_FLOOR:
        pred["predicted_class"] = "uncertain"

    pred["warning"] = WARNING_TEXT
    pred["guardrail_errors"] = errors
    return pred
