"""Sous-processus unique du smoke test MedGemma.

Charge MedGemma **une seule fois** (via le cache `_PIPE` de `src.inference_medgemma`)
et produit en un seul passage les deux artefacts dont a besoin la suite d'intégration :

1. `sample_pred` : une prédiction complète (garde-fous appliqués) pour vérifier le
   contrat schéma / warning ;
2. `summary` : l'évaluation jouet sur le dataset synthétique (réutilise le même
   pipeline en cache → aucun rechargement du modèle).

Le préfixe `_` du nom de fichier empêche pytest de le collecter comme module de test.

Pourquoi un sous-processus ? torch/numpy crashent dans le process pytest sur Windows
(état FPE modifié par pytest) ; on isole donc l'inférence dans un process frais. Le but
de ce script est qu'il n'y ait **qu'un seul** de ces process par run de smoke test.

Usage : python _medgemma_smoke_subprocess.py <out_dir> <db_path> <image_path>
Sortie : une ligne JSON sur stdout {sample_pred, summary, out_dir, db_path}.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

# `src.inference_medgemma` pose KMP_DUPLICATE_LIB_OK avant d'importer torch (cf. note
# OpenMP dans le module). On l'importe donc en premier.
from src.inference_medgemma import medgemma_predict
from src.guardrails import apply_safety_guardrails
import run_evaluation  # eval/run_evaluation.py : réutilise read_cases / run / write_csv


def main() -> None:
    out_dir = Path(sys.argv[1])
    db_path = Path(sys.argv[2])
    image_path = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Premier appel → charge MedGemma et met `_PIPE` en cache. Sert aussi de
    #    prédiction d'échantillon pour le contrat schéma / garde-fous.
    sample_pred = apply_safety_guardrails(medgemma_predict(image_path, mode="baseline"))

    # 2) Évaluation jouet : `run` rappelle `medgemma_predict` pour chaque cas → cache
    #    hit, le modèle n'est PAS rechargé. Mêmes sorties CSV/JSON que run_evaluation.main().
    cases = run_evaluation.read_cases(ROOT / "data" / "synthetic_cases.csv")
    rows, metrics = run_evaluation.run("baseline", db_path, cases, backend="medgemma")
    run_evaluation.write_csv(out_dir / "baseline_predictions.csv", rows)
    (out_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = [{"mode": "baseline", **metrics}]
    run_evaluation.write_csv(out_dir / "before_after_summary.csv", summary)

    print(json.dumps({
        "sample_pred": sample_pred,
        "summary": summary,
        "out_dir": str(out_dir),
        "db_path": str(db_path),
    }))


if __name__ == "__main__":
    main()
