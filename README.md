# 🌾 Moussa Kouyaté — Application Streamlit Soninké

Application de dialogue en soninké basée sur **GuppyLM**, un Transformer entraîné
sur 10 000+ paires de dialogues avec le vocabulaire SIL (3 231 mots authentiques).

## Structure du projet

```
moussa_streamlit_app/
├── app.py                        # Application Streamlit principale
├── model.py                      # Tokeniseur + Architecture GuppyLM
├── requirements.txt              # Dépendances Python
├── README.md                     # Ce fichier
├── .streamlit/
│   └── config.toml               # Thème sombre personnalisé
└── guppylm_moussa_finetuned.pt   # ← AJOUTER après entraînement
    (ou guppylm_moussa_best.pt)
```

## Installation et lancement

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Copier le modèle entraîné dans ce dossier
cp /chemin/vers/guppylm_moussa_finetuned.pt .

# 3. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur **http://localhost:8501**

## Déploiement sur Streamlit Cloud

1. Pousser ce dossier sur GitHub
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter le dépôt et choisir `app.py`
4. Cliquer **Deploy** ✅

> ⚠️ Si le fichier `.pt` dépasse 100 MB, hébergez-le sur Hugging Face Hub
> et chargez-le dynamiquement avec `huggingface_hub.hf_hub_download()`.

## Personnages et catégories

| Catégorie    | Exemples de questions                              |
|--------------|----------------------------------------------------|
| Salutations  | `Haayi, Moussa!`                                   |
| Agriculture  | `Yillen wa naxa ba ke xaaxo?`                      |
| Élevage      | `An nanu wa laafin ba?`                            |
| Marché       | `Saxanen wa naxa ba ke fane?`                      |
| Météo        | `Kanmen wa riini ba ke siine?`                     |
| Famille      | `An denbaya wa laafin ba?`                         |
| Religion     | `Alla wa an teen deema ba?`                        |
| Philosophie  | `Manni sagesse wa an maxa teen golle ba?`          |
