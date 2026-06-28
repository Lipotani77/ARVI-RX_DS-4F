from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS = Path(__file__).resolve().parent / "_medgemma_smoke_subprocess.py"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def medgemma_smoke_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Charge MedGemma **une seule fois** pour toute la suite d'intégration.

    Les deux tests ci-dessous partagent ce fixture : un unique sous-processus charge
    le modèle, fait la prédiction d'échantillon ET l'évaluation jouet (cache `_PIPE`
    réutilisé). Évite le double chargement (~16 Go transitoires) qui saturait la VRAM.
    Sous-processus isolé car torch/numpy crashent dans le process pytest sur Windows.
    """
    work = tmp_path_factory.mktemp("medgemma_smoke")
    out_dir = work / "outputs"
    db_path = work / "medical_ai_evidence.sqlite"
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"

    result = subprocess.run(
        [sys.executable, str(SUBPROCESS), str(out_dir), str(db_path), str(image_path)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # Le modèle imprime ses logs/barres de progression sur stderr ; notre JSON est la
    # dernière ligne de stdout.
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_medgemma_prediction_schema_warning_and_guardrails(medgemma_smoke_result: dict) -> None:
    from src.guardrails import WARNING_TEXT, validate_prediction

    pred = medgemma_smoke_result["sample_pred"]
    valid, errors = validate_prediction(pred)
    assert valid, errors
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert pred["warning"] == WARNING_TEXT


def test_medgemma_evaluation_command_runs_and_preserves_warning_contract(
    medgemma_smoke_result: dict,
) -> None:
    summary = medgemma_smoke_result["summary"]
    assert {row["mode"] for row in summary} == {"baseline"}
    assert all(row["json_valid_rate"] >= 0.9 for row in summary)
    assert all(row["warning_rate"] == 1.0 for row in summary)
    assert (Path(medgemma_smoke_result["out_dir"]) / "before_after_summary.csv").exists()
    assert Path(medgemma_smoke_result["db_path"]).exists()
