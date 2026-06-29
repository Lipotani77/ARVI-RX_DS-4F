# Guide Utilisateur - ARVI-RX (Assistant Radiologue Virtuel)

Bienvenue dans l'interface de démonstration du projet ARVI-RX (Étape S5). Cette application web permet d'analyser des radiographies thoraciques frontales grâce à un modèle d'Intelligence Artificielle (MedGemma-4B-IT).

## 1. Démarrer l'application

Pour lancer l'application sur votre machine locale, ouvrez un terminal à la racine du projet et exécutez la commande suivante :
```bash
streamlit run app/streamlit_app.py
```
Une fenêtre s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse `http://localhost:8501`.

## 2. Réaliser une nouvelle analyse

1. **Uploader une image** : Cliquez sur le bouton "Browse files" ou glissez-déposez une radiographie au format `.png`, `.jpg` ou `.jpeg`. Vous pouvez utiliser les images de test fournies dans le dossier `data/sample_images/`.
2. **Choisir le prompt (Configuration)** : Dans la barre latérale, sélectionnez la version du prompt à utiliser pour l'inférence. Le mode **Advanced** est fortement recommandé pour garantir un format JSON valide et limiter les hallucinations.
3. **Lancer l'analyse** : Cliquez sur le bouton d'analyse.
4. **Prétraitement** : L'application effectuera un prétraitement visuel (Anonymisation DICOM simulée et redimensionnement) avant d'appeler l'IA.
5. **Résultats** : Une fois l'analyse terminée, la page affichera le diagnostic suggéré, le niveau de confiance, les preuves visuelles, les recommandations médicales associées, ainsi que l'avertissement de sécurité.

## 3. Consulter l'historique

Toutes les analyses que vous effectuez sont automatiquement sauvegardées dans une base de données locale (SQLite).
L'onglet Historique permet de lister les précédentes inférences avec leurs résultats et de tracer l'évolution des diagnostics.

## 4. Précautions d'usage

⚠️ **Ceci est un prototype pédagogique.**
L'application et le modèle IA sous-jacent ne sont pas destinés au diagnostic médical. Toute analyse générée par ce système doit impérativement être validée par un professionnel qualifié. L'intégration de la fonctionnalité "Incertain" sert justement de garde-fou en cas d'image floue ou atypique.
