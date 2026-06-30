from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

# Double runtime OpenMP (MKL + PyTorch) sous Anaconda → OMP: Error #15. On l'autorise
# avant tout import de numpy/torch. Cf. note dans src/inference_medgemma.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails, validate_prediction
from src.metrics import summarize_metrics, CLASSES
from src.database import insert_run, init_db


def read_cases(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def build_rsna_cases(limit: int = 20) -> list[dict]:
    """Construit la liste de cas depuis les Images/Masks RSNA.

    Aucun CSV de labels n'accompagne RSNA_Pneumonia : on dérive la vérité-terrain
    du masque de segmentation apparié (même nom de fichier). Masque tout-noir →
    `normal`, masque avec au moins un pixel non nul → `suspected_opacity`.
    Même format de dict que `read_cases` (clés `case_id`, `image_path`, `label`).
    """
    img_dir = ROOT / 'data' / 'RSNA_Pneumonia' / 'Images'
    mask_dir = ROOT / 'data' / 'RSNA_Pneumonia' / 'Masks'
    cases = []
    for img in sorted(img_dir.glob('*.png'))[:limit]:
        mask = np.array(Image.open(mask_dir / img.name).convert('L'))
        label = 'suspected_opacity' if mask.any() else 'normal'
        cases.append({
            'case_id': img.stem,
            'image_path': str(img.relative_to(ROOT)).replace('\\', '/'),
            'label': label,
        })
    return cases


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def write_confusion_csv(path: Path, matrix: dict[str, dict[str, int]],
                        classes: list[str] = CLASSES) -> None:
    """Écrit la matrice de confusion : lignes = vérité-terrain, colonnes = prédiction."""
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['true\\pred', *classes])
        for t in classes:
            w.writerow([t, *(matrix[t][p] for p in classes)])


def run(mode: str, db_path: Path, cases: list[dict], backend: str = 'toy',
        device: str | None = None) -> tuple[list[dict], dict]:
    rows = []
    init_db(db_path)
    if backend == 'medgemma':
        # Import paresseux : ne charge torch/transformers que si on en a besoin.
        from src.inference_medgemma import medgemma_predict
        predict = lambda p: medgemma_predict(p, mode=mode, device=device)
    else:
        predict = lambda p: toy_predict(p, mode=mode)
    for case in cases:
        image_path = ROOT / case['image_path']
        pred = apply_safety_guardrails(predict(image_path))
        valid, errors = validate_prediction(pred)
        # json_valid = schéma conforme ET JSON réellement parsable. Le parser MedGemma
        # expose `json_valid` (False sur sortie tronquée même si la classe a été récupérée) :
        # on le combine pour que json_valid_rate reflète la troncature, sans masquer le
        # gain d'accuracy. Backend `toy` : pas de clé → défaut True → comportement inchangé.
        json_valid = valid and pred.get('json_valid', True)
        row = {
            'case_id': case['case_id'],
            'label': case['label'],
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'json_valid': json_valid,
            'warning': pred.get('warning', ''),
            'latency_ms': pred.get('latency_ms', 0),
            'guardrail_errors': ';'.join(errors),
        }
        rows.append(row)
        insert_run(db_path, case['case_id'], str(image_path), pred)
    metrics = summarize_metrics(rows)
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['toy', 'baseline', 'improved', 'advanced'], default='toy')
    parser.add_argument('--backend', choices=['toy', 'medgemma'], default='toy')
    parser.add_argument('--dataset', choices=['synthetic', 'rsna'], default='synthetic')
    parser.add_argument('--limit', type=int, default=20,
                        help='nombre d\'images RSNA (ignoré pour le dataset synthetic)')
    parser.add_argument('--device', default=None,
                        help='cuda | cpu | None (auto-détection) — passé à medgemma_predict')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'eval' / 'outputs')
    parser.add_argument('--db-path', type=Path, default=ROOT / 'medical_ai_evidence.sqlite')
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'rsna':
        cases = build_rsna_cases(args.limit)
    else:
        cases = read_cases(ROOT / 'data' / 'synthetic_cases.csv')

    modes = ['baseline', 'improved'] if args.mode == 'toy' else [args.mode]
    if args.backend == 'medgemma':
        # MedGemma ne définit que le prompt `baseline` (cf. PROMPTS dans
        # src/inference_medgemma.py) → on filtre les modes indisponibles pour
        # éviter un KeyError, et on prévient l'utilisateur le cas échéant.
        from src.inference_medgemma import PROMPTS
        available = [m for m in modes if m in PROMPTS]
        missing = [m for m in modes if m not in PROMPTS]
        if missing:
            print(f"[backend=medgemma] prompts indisponibles, ignorés : {missing}")
        modes = available or ['baseline']

    summary = []
    for mode in modes:
        rows, metrics = run(mode, args.db_path, cases, backend=args.backend, device=args.device)
        write_csv(out_dir / f'{mode}_predictions.csv', rows)
        (out_dir / f'{mode}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        write_confusion_csv(out_dir / f'{mode}_confusion_matrix.csv', metrics['confusion_matrix'])
        summary.append({'mode': mode, **metrics})
    write_csv(out_dir / 'before_after_summary.csv', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
