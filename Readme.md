# 📊 Projet IA & Règles : Détection de Points Fantômes

Ce projet universitaire de Master 2 MIAGE (parcours MIAGE MIXTES) est divisé en deux parties sous forme de notebooks :

- **Notebook 1 - Préparation des données & création du modèle IA**  
- **Notebook 2 - Détection de points fantômes via IA et via règles métier**

## 🧠 Objectif

L'objectif est de détecter les **points fantômes** dans un jeu de données en comparant deux approches :  
- Une **approche basée sur un modèle d'intelligence artificielle (IA)**  
- Une **approche basée sur un ensemble de règles expertes définies manuellement**

## 📁 Structure du projet

```
📦 NEJMA_SMATTI_POC_2025/
├── Readme.md
├── requirements.txt
├── App/
│   ├── config.py
│   ├──  autoencoder.py
│   ├── dataset.py
│   └── visualisation.py
│
│── DataPrep_and_Model.ipynb
└── Detection_Method.ipynb

```

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git https://github.com/nejmas/NEJMA_SMATTI_POC_2025.git
cd NEJMA_SMATTI_POC_2025.git
```

### 2. Créer un environnement virtuel

**Python 3.8+** doit être installé.

```bash
python -m venv venv
```

### 3. Activer l’environnement virtuel

- **Sur Windows** :

```bash
venv\Scripts\activate
```

- **Sur macOS/Linux** :

```bash
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

> Le fichier `requirements.txt` contient toutes les bibliothèques nécessaires : `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, etc.

## 📓 Utilisation des notebooks

Lance Jupyter Notebook pour accéder aux deux carnets :

```bash
jupyter notebook
```
