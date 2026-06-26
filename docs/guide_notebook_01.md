# Guide — Lancer le notebook `01_baseline_vlm.ipynb`

Ce guide explique **pas à pas** comment configurer votre machine pour exécuter le premier notebook
([notebooks/01_baseline_vlm.ipynb](../notebooks/01_baseline_vlm.ipynb)).

Le notebook charge le modèle **MedGemma** (`google/medgemma-4b-it`) depuis Hugging Face, analyse une
radiographie thoracique et renvoie un JSON. Suivez les étapes dans l'ordre.

---

## ⚠️ À savoir avant de commencer

- **C'est un prototype pédagogique**, pas un dispositif médical. Ne l'utilisez jamais pour un vrai diagnostic.
- **MedGemma est un modèle « gated »** sur Hugging Face : il faut un compte, accepter les conditions d'accès,
  et s'authentifier. Sans ça, le téléchargement échoue (étape 4).
- **Le modèle est lourd** (~8 Go en bfloat16). Prévoyez :
  - Idéalement un **GPU NVIDIA avec ≥ 8 Go de VRAM** (fp16, rapide, < 10 s/image).
  - **Petit GPU (4–6 Go)** : le code charge automatiquement le modèle en **4‑bit** (~3 Go) pour qu'il tienne. Mesuré sur RTX 3050 4 Go : **~20 s/image** (au repos), plus lent si le GPU est partagé avec le bureau.
  - Sinon le **CPU** fonctionne aussi (auto-détecté), mais c'est **très lent** : **~7–8 min par image** (mesuré sur i7‑12650H).
  - Au moins **~10 Go d'espace disque libre** pour le téléchargement des poids.

> ⏱️ **Latence & objectif < 10 s** : voir l'analyse mesurée dans
> [docs/latence_et_materiel.md](latence_et_materiel.md). En résumé, < 10 s avec le JSON
> complet suppose un GPU ≥ 8 Go ; sur 4 Go on est à ~20 s, documenté comme limite matérielle.
> Le 4‑bit requiert le paquet `bitsandbytes` (déjà dans `requirements.txt`).

---

## 1. Prérequis

- **Python 3.10 à 3.13** installé ([python.org](https://www.python.org/downloads/)).
- **Git** installé pour cloner le dépôt.
- Un **compte Hugging Face** gratuit ([huggingface.co/join](https://huggingface.co/join)).

Vérifiez votre version de Python :

```bash
python --version
```

---

## 2. Récupérer le projet

```bash
git clone <URL_DU_DEPOT>
cd ARVI-RX_DS-4F
```

(Si vous avez déjà le dossier, placez-vous simplement dedans.)

---

## 3. Créer l'environnement virtuel et installer les dépendances

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On installe aussi les outils pour exécuter un notebook :

```bash
pip install jupyter ipykernel
```

### Avec Anaconda / Miniconda (alternative au `venv`)

Si vous utilisez la **distribution Anaconda** (ou Miniconda), n'utilisez **pas** `python -m venv`.
Créez plutôt un environnement `conda` dédié. À faire dans un terminal **Anaconda Prompt**
(Windows) ou n'importe quel terminal où `conda` est disponible :

```bash
conda create -n arvi-rx python=3.11 -y
conda activate arvi-rx
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter ipykernel
```

> ⚠️ **Ne mélangez pas** `conda` et `venv` : choisissez l'un ou l'autre. Si vous êtes sur Anaconda,
> utilisez l'environnement `arvi-rx` ci-dessus partout où le guide mentionne `.venv`.

Enregistrez ensuite l'environnement comme kernel Jupyter (utile pour le sélectionner à l'étape 5) :

```bash
python -m ipykernel install --user --name arvi-rx --display-name "Python (arvi-rx)"
```

> 💡 **Astuce GPU** : `pip install torch` installe par défaut une version CPU.
> Si vous avez un GPU NVIDIA, installez la version CUDA depuis
> [pytorch.org/get-started](https://pytorch.org/get-started/locally/) pour profiter de l'accélération.

---

## 4. Se connecter à Hugging Face (indispensable)

MedGemma est un modèle à accès contrôlé. Trois sous-étapes :

1. **Créer un compte** sur [huggingface.co](https://huggingface.co/join).
2. **Accepter les conditions d'accès** du modèle : allez sur
   [huggingface.co/google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
   et cliquez sur le bouton pour demander/valider l'accès (acceptation immédiate après avoir rempli le formulaire Google).
3. **Créer un token** d'accès : [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   → *New token* → type **Read** → copiez la valeur.

> 🔑 **Important — conservez votre token quelque part de sûr** (gestionnaire de mots de passe, note
> privée…). Hugging Face ne vous le **réaffichera jamais** après la création : si vous le perdez, vous
> devrez en regénérer un nouveau. Vous en aurez besoin si vous changez de machine, réinstallez
> l'environnement, ou perdez l'authentification locale.
>
> ❌ **Ne le partagez pas et ne le committez jamais** dans le dépôt Git (ni dans une cellule du notebook).
> C'est un secret personnel équivalent à un mot de passe.

Puis authentifiez-vous en local :

```bash
pip install huggingface_hub
huggingface-cli login
```

Collez votre token quand il est demandé. (Le token est mémorisé, à faire **une seule fois** par machine.)

> Le code détecte automatiquement le GPU ou le CPU — vous n'avez rien à configurer côté device.

---

## 5. Lancer le notebook

```bash
jupyter notebook
```

Dans la fenêtre qui s'ouvre dans le navigateur :

1. Ouvrez `notebooks/01_baseline_vlm.ipynb`.
2. Vérifiez que le **kernel** sélectionné est bien celui du `.venv`
   (menu *Kernel → Change kernel* si besoin).
3. Exécutez les cellules **dans l'ordre** (`Shift + Entrée`).

> Alternative : ouvrez le notebook directement dans **VS Code** (extension Python + Jupyter),
> et sélectionnez l'interpréteur du `.venv` en haut à droite.

---

## 6. Ce que fait le notebook

| Cellule | Rôle |
|---|---|
| 1 | Configure l'environnement (correctif Windows `KMP_DUPLICATE_LIB_OK`) et importe les fonctions du projet. |
| 2 | Analyse une **image synthétique** (`data/sample_images/...`) avec `medgemma_predict` + garde-fous. |
| 3 | Affiche le résultat JSON (classe, confiance, observations, justification, limites, warning). |
| 4-6 | (Optionnel) Teste une image du dataset **RSNA Pneumonia**. |

**Le premier appel télécharge le modèle (~8 Go)** : soyez patient, c'est normal que la première exécution
soit longue. Les fois suivantes, le modèle est mis en cache localement.

---

## 7. Note sur les cellules RSNA (4 à 6)

Ces cellules pointent vers les images du dataset RSNA Pneumonia. **Bonne nouvelle : ces images sont
déjà incluses dans le dépôt Git** — vous n'avez **rien à télécharger ni à configurer** de ce côté-là.

👉 Après avoir cloné le projet (étape 2), vous pouvez exécuter **toutes les cellules (1 à 6)** directement.

---

## 8. Problèmes fréquents

| Erreur | Cause / Solution |
|---|---|
| `OSError ... gated repo` / `401` | Accès non validé ou token absent → refaites l'**étape 4**. |
| `OMP: Error #15 ... libiomp5md.dll` | Déjà géré par la 1ʳᵉ cellule (`KMP_DUPLICATE_LIB_OK`). Exécutez-la en premier. |
| Téléchargement très long / RAM saturée | Normal sur CPU. Fermez les autres applications ; le modèle fait ~8 Go. |
| `ModuleNotFoundError: src` | Lancez le notebook **depuis le dossier `notebooks/`** (le code fait `sys.path.append('..')`). |
| `CUDA out of memory` | GPU avec trop peu de VRAM → le code bascule sur CPU si < 8 Go. Fermez les autres process GPU. |
| Mauvais kernel | Sélectionnez le kernel du `.venv` (étape 5). |

---

## Récapitulatif express

```bash
# 1. Environnement
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
pip install jupyter ipykernel huggingface_hub

# 2. Authentification Hugging Face (après avoir accepté l'accès au modèle)
huggingface-cli login

# 3. Lancer
jupyter notebook
# → ouvrir notebooks/01_baseline_vlm.ipynb et exécuter les cellules (1 à 6)
```

Bon courage ! 🩻
