---
name: medgemma-vram-contention
description: GPU 4 Go — un seul consommateur de MedGemma à la fois ; sinon OSError 1455 au chargement
metadata:
  type: project
---

La machine de dev n'a que **4 Go de VRAM** (RTX 3050) et ~32 Go RAM. MedGemma‑4b se charge en lisant ~8 Go de poids en RAM (CPU) avant de quantifier en 4‑bit vers le GPU (~3,38 Go de pic VRAM).

**Piège récurrent** : si un autre process Python tient déjà le modèle (typiquement le **kernel Jupyter du notebook 01** laissé ouvert), il monopolise RAM **et** VRAM. Le chargement suivant (app Streamlit, autre run) échoue alors avec :
`OSError: Le fichier de pagination est insuffisant... (os error 1455)` — la **limite de commit** (RAM + pagefile) est atteinte, et/ou le GPU bascule sur CPU faute de VRAM libre (`_resolve_device` exige ≥ 3,2 Go libres, cf. `MIN_VRAM_BYTES_4BIT`).

Diagnostic / résolution (cf. [[python-env]]) :
- `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` → qui tient la VRAM.
- `Get-Process python | sort WS -desc` → qui tient la RAM (un kernel chargé ≈ 11 Go WS).
- `Stop-Process -Id <pid> -Force` sur les kernels/process orphelins, puis **relancer le run au propre** (un process frais reprend toute la VRAM).

**Règle** : un seul consommateur du modèle à la fois. Ne pas garder le notebook 01 et l'app Streamlit chargés en parallèle. Vérifier ≥ 3,2 Go VRAM libres **avant** de lancer. Le pagefile (~17 Go) n'est PAS le problème ; c'est la saturation par process zombie.
