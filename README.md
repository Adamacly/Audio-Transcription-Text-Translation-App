# 📝 Audio Transcription & Translation App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Deployed-yellow.svg)

Une application web pour transcrire des fichiers audio en français et traduire le texte en anglais, basée sur Whisper (OpenAI) et NLLB (Meta).

## 🚀 Démo en ligne

L'application est déployée sur Hugging Face Spaces :  
🔗 [https://huggingface.co/spaces/AdamaAdam/Audio-Transcription-And-Text-Translation-App](https://huggingface.co/spaces/AdamaAdam/Audio-Transcription-And-Text-Translation-App)

## ✨ Fonctionnalités

- **Transcription audio** : Conversion de parole en texte (FR)
- **Traduction automatique** : Traduction français → anglais (EN) de qualité
- **Multiples sources d'entrée** :
  - Upload de fichiers audio (WAV, MP3)
  - Enregistrement direct via microphone
- Interface simple et intuitive avec Gradio

## 🛠 Installation locale

#### 1. Cloner le dépôt :
```bash
git clone https://github.com/Adamacly/Audio-Transcription-Text-Translation-App.git
cd Audio-Transcription-Text-Translation-App
```

#### 2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate
venv/Scripts/activate     
```

#### 3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Lancer l'application :
```bash
python app.py
```