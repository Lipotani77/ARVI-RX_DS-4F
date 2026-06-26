# Latence & matériel — analyse mesurée

> Note de performance pour le rapport / la soutenance.
> Objectif du sujet : **latence < 10 s** par image (cf. cibles d'évaluation).
> **Verdict : non atteignable avec le JSON complet sur le matériel testé — documenté comme limite matérielle.**

## Matériel de référence (machine de dev)

| Composant | Détail |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop — **4 Go** VRAM (Ampere) |
| CPU | Intel Core i7‑12650H (10 cœurs / 16 threads, Alder Lake, **pas d'unité bf16 native**) |
| RAM | 32 Go |
| Modèle | `google/medgemma-4b-it` (~4 Md paramètres, VLM) |

## Latences mesurées (1 image, `medgemma_predict`)

| Configuration | Latence / image | < 10 s ? | JSON |
|---|---|---|---|
| CPU bfloat16 (auto-détection par défaut) | **~450 000 ms** (~7,5 min) | ❌ (×45) | complet |
| GPU 4‑bit NF4, GPU au repos, process frais | **~22 000–26 000 ms** | ❌ | complet ✅ |
| GPU 4‑bit NF4, GPU partagé avec le bureau Windows | **~50 000–75 000 ms** (erratique) | ❌ | complet |
| GPU 4‑bit, sortie tronquée à ~46 tokens | ~9 000 ms | ✅ | **JSON incomplet** ❌ |

Le passage CPU → GPU 4‑bit apporte un gain **×17 à ×20**. C'est la bonne optimisation ; mais le dernier facteur ~2 pour franchir les 10 s se heurte à deux murs physiques décrits ci‑dessous.

## Pourquoi < 10 s est impossible ici (diagnostic chiffré)

La latence d'un VLM génératif ≈ **prefill (coût fixe) + N_tokens × coût/token**. Mesure sur GPU 4‑bit (même image, `max_new_tokens` variable) :

| Tokens générés | Latence | Longueur sortie |
|---|---|---|
| 8 | 2 656 ms | 17 car. |
| 32 | 6 844 ms | 84 car. |
| 64 | 11 914 ms | 197 car. |
| 96 | 17 060 ms | 290 car. |
| 128 | 25 225 ms | 413 car. |

Régression linéaire : **prefill image ≈ 1,2 s**, **décodage ≈ 188 ms/token (~5,3 tok/s)** en 4‑bit sur cette 3050.

**Mur n°1 — budget de tokens.** 10 000 ms ⟹ ~46 tokens générables. Or le **JSON imposé par le contrat** (7 champs + justification « 2 à 4 phrases ») nécessite **~90–130 tokens**, soit **~17–25 s incompressibles**. À 46 tokens, on ne peut même pas écrire la structure JSON complète → on briserait le schéma partagé (inference/guardrails/API/éval).

**Mur n°2 — VRAM trop juste.** Le modèle 4‑bit occupe un **pic de 3,38 Go** ; avec ~0,8 Go déjà pris par le bureau Windows, il ne reste quasi **aucune marge** (0 Go libre observé). Sous Windows (WDDM), une VRAM saturée déverse silencieusement vers la **RAM partagée** (beaucoup plus lente) → latence **erratique** (25 s au repos, 50–75 s sous charge).

## Quantification 4‑bit : ce qui a été implémenté

Dans [`src/inference_medgemma.py`](../src/inference_medgemma.py) :

- **GPU** : chargement en **4‑bit NF4** (`BitsAndBytesConfig`, `bnb_4bit_compute_dtype=bfloat16`, double quantization) → ~3 Go au lieu de ~8 Go, donc **tient dans 4 Go**.
- **Seuil VRAM** abaissé : `MIN_VRAM_BYTES_4BIT = 3,2 Go` (avant : 8 Go, qui forçait le repli CPU sur les petits GPU).
- **CPU** : bfloat16 par défaut, précision et threads réglables (voir knobs ci‑dessous).
- **Instrumentation** : `latency_ms`, `model_name`, `prompt_version` ajoutés à chaque prédiction (le chrono **exclut** le chargement unique du modèle).
- Dépendance ajoutée : `bitsandbytes>=0.43` (pour la quantification GPU).

### Réglages disponibles (variables d'environnement)

| Variable | Effet | Quand l'utiliser |
|---|---|---|
| `device="cuda"` (argument) | force le GPU 4‑bit | recommandé sur cette machine |
| `ARVI_CPU_PRECISION=float32` | CPU en float32 (~16 Go RAM) au lieu de bf16 émulé | CPU only, si ≥ ~20 Go RAM libre |
| `ARVI_CPU_THREADS=10` | limite les threads CPU | mesurer P‑cores vs tous les threads |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | réduit la fragmentation VRAM | marge VRAM très serrée |

## Comment atteindre réellement < 10 s (hors périmètre actuel)

1. **GPU ≥ 8 Go** : le modèle tient en **fp16** (pas de surcoût de déquantification 4‑bit) → décodage **~3–5× plus rapide** → JSON complet sous 10 s. *Voie la plus simple et fiable.*
2. **Alléger la sortie générée** : ne demander au modèle que les champs cliniques (`predicted_class`, `confidence`, `visual_evidence`, justification courte) et remplir par code les champs constants/dérivables (`warning` déjà constant, `image_quality` via `preprocessing`, `limitations` par gabarit). Tombe vers ~10–13 s, mais **modifie la baseline** (justification plus courte).
3. **Modèle plus petit** (Gemma 4 E2B/E4B) : plus rapide, mais sort du modèle baseline recommandé.

## Conclusion pour le rapport

Sur une machine grand public à **4 Go de VRAM**, MedGemma‑4B avec le contrat JSON complet tourne en **~20 s/image** (GPU 4‑bit) — non‑conforme à la cible des 10 s, qui suppose implicitement un GPU ≥ 8 Go. Ce n'est **pas un défaut du prototype** mais une **contrainte matérielle mesurée et tracée** : le prototype reste correct, prudent et conforme au schéma. C'est exactement le type de limite que le sujet demande de **documenter plutôt que de masquer**.
