---
name: python-env
description: How to run Python/pytest/compile in this project — no python on PATH, use the conda env
metadata:
  type: project
---

Il n'y a **aucun `python` réel sur le PATH** (les `python.exe`/`python3.exe` dans `WindowsApps` sont des stubs Microsoft Store, et `py` n'existe pas). Il n'y a pas non plus de `.venv` dans le repo.

L'environnement de travail est **conda** : `conda env list` montre `base` et **`arvi-rx`**. Pour tout `python` / `pytest` / `compileall`, utiliser directement l'interpréteur de l'env :

`C:\Users\Utilisateur\anaconda3\envs\arvi-rx\python.exe`

(Python 3.11.15 ; `streamlit`, `torch`, `transformers` y sont installés ; `_resolve_device()` renvoie `cuda` → RTX 3050 utilisée en 4-bit, cf. [[latence-medgemma]].)
