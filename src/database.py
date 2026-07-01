from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def connect(db_path: str | Path = "medical_ai_evidence.sqlite") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = "medical_ai_evidence.sqlite") -> None:
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit(); conn.close()


def insert_run(db_path: str | Path, case_id: str, image_path: str, prediction: dict) -> None:
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO runs(case_id, image_path, model_name, prompt_version, prediction_json, predicted_class, confidence, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            image_path,
            prediction.get("model_name"),
            prediction.get("prompt_version"),
            json.dumps(prediction, ensure_ascii=False),
            prediction.get("predicted_class"),
            float(prediction.get("confidence", 0.0)),
            int(prediction.get("latency_ms", 0)),
        ),
    )
    conn.commit(); conn.close()


def insert_case_result(db_path: str | Path, row: dict, prediction: dict) -> None:
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO case_results(
            case_id, image_path, label, predicted_class, confidence,
            safety_predicted_class, safety_confidence, json_valid, warning,
            latency_ms, guardrail_errors, model_name, prompt_version, prediction_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.get("case_id"),
            row.get("image_path", ""),
            row.get("label"),
            row.get("predicted_class"),
            float(row.get("confidence", 0.0)),
            row.get("safety_predicted_class", ""),
            float(row.get("safety_confidence", 0.0) or 0.0),
            int(bool(row.get("json_valid", False))),
            row.get("warning", ""),
            int(row.get("latency_ms", 0)),
            row.get("guardrail_errors", ""),
            prediction.get("model_name"),
            prediction.get("prompt_version"),
            json.dumps(prediction, ensure_ascii=False),
        ),
    )
    conn.commit(); conn.close()
