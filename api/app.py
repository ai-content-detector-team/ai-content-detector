#══════════════════════════════════════════════════════════════════════════════════════
#                              IMPORT DES LIBRAIRIES
#══════════════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union
import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import mlflow
import spacy
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer



#══════════════════════════════════════════════════════════════════════════════════════
#                           DESCRIPTION COMPLETE DE L'API
#══════════════════════════════════════════════════════════════════════════════════════

description = """
## AI Review Detector API

L'**AI Review Detector API** est une solution professionnelle de détection automatique des avis générés par intelligence artificielle.

### 🎯 Objectif

Cette API permet d'identifier si un avis client ou une critique a été rédigé par un humain ou généré par une intelligence artificielle, 
aidant ainsi les entreprises à maintenir l'authenticité de leurs plateformes d'évaluation.

### 🔬 Modèle de détection

Notre système repose sur un modèle **XGBoost** entraîné sur un vaste corpus de textes authentiques et générés par IA.

**Performances du modèle sur de nouvelles données :**
- **Accuracy** : XX.XX%
- **Precision** : XX.XX%
- **Recall** : XX.XX%
- **F1-Score** : XX.XX%

**Dataset d'entraînement :**
- XXX avis étiquetés (XXX humains / XXX IA)
- Source : Kaggle (liens vers les datasets)
- Langue : Anglais (uniquement, non?)
- Taille variable : De courts commentaires à des essais détaillés

### 💡 Cas d'usage

- **E-commerce** : Validation de l'authenticité des avis produits
- **Hôtellerie** : Détection de faux avis sur les plateformes de réservation  
- **Modération** : Filtrage automatique des contenus générés automatiquement par IA
- **Analyse qualitative** : Audit de bases de données d'avis existantes

### 🚀 Fonctionnalités

Cette API propose deux modes d'utilisation :

1. **Analyse unitaire** : Détection pour un seul texte
2. **Traitement par lot** : Analyse de jusqu'à 10,000 textes simultanément pour un traitement efficace

### 📊 Format des données

**Entrée :** Texte brut (sans limite de longueur)  
**Sortie :** Classification binaire (1 = IA ou 0 = Humain)

### ⚡ Performance

Temps de réponse moyen : < XXXms par texte
"""

app = FastAPI(
    title="AI Review Detector API",
    description=description,
    version="1.0.0",
    contact={
        "name": "AI Review Detector Team",
        "email": "contact@aireviewdetector.com",
    }
)



#══════════════════════════════════════════════════════════════════════════════════════
#                                MODELES DE DONNEES
#══════════════════════════════════════════════════════════════════════════════════════

#Modèle de données pour une requête unitaire
class TextInput(BaseModel):
    text: str = Field(
        ...,
        description="Texte de l'avis à analyser",
        example="This product exceeded my expectations! The quality is outstanding and delivery was super fast"
    )

#Modèle de données pour une requête batch
class BatchTextInput(BaseModel):
    texts: Union[list[str], list[dict]] = Field(
        ...,
        description="Liste de textes à analyser (maximum 10,000)",
        max_length=10000,
        example=[
            "This product exceeded my expectations! The quality is outstanding.",
            {"text": "I really enjoyed my stay at this hotel. The staff was very friendly."},
            "The food was delicious and the service was impeccable."
        ]
    )

#Modèle de données pour une réponse unitaire
class PredictionResponse(BaseModel):
    is_ai_generated: int = Field(..., description="1 si généré par l'IA, 0 si écrit par un humain")
    message: str = Field(..., description="Message explicatif du résultat")

#Modèle de données pour une réponse batch
class BatchPredictionResponse(BaseModel):
    predictions: list[int] = Field(..., description="Liste des prédictions (1 = IA, 0 = Humain)")


#══════════════════════════════════════════════════════════════════════════════════════
#                               CONFIGURATION GLOBALE
#══════════════════════════════════════════════════════════════════════════════════════

#Variables globales pour les ressources NLP
nlp = None
stop_en = None
tokenizer = None
model = None

#MLflow
MLFLOW_TRACKING_URI = "https://name-space_name.hf.space"
MODEL_URI = "models:/name_model/number_version"

#Configuration
STOPWORDS_LANGUAGE = "english"
BERT_MODEL_NAME = "bert-base-uncased"

#Listes de ponctuations et connecteurs
PUNCT_LIST = ['!', '?', ',', '.', ';', ':', '"', "'", '(']
ELLIPSIS_TOKEN = '...'

ALL_POS_TAG = [
    "DET", "VERB", "SCONJ", "AUX", "PART", "CCONJ",
    "ADV", "ADJ", "ADP", "PROPN", "PRON", "NOUN", "NUM"
]

connectives = {
    'addition': {'and', 'also', 'furthermore', 'moreover', 'in addition', 'besides', 'as well', 'what is more',
                 'not only... but also', 'similarly', 'likewise'},
    'contrast': {'but', 'however', 'on the other hand', 'nevertheless', 'nonetheless', 'yet', 'still',
                 'even so', 'although', 'though', 'whereas', 'while', 'in contrast', 'conversely'},
    'cause': {'because', 'since', 'as', 'due to', 'owing to', 'thanks to', 'considering that', 'for the reason that'},
    'consequence': {'so', 'therefore', 'thus', 'hence', 'as a result', 'consequently', 'accordingly', 'for this reason'},
    'concession': {'although', 'even though', 'though', 'while', 'granted that', 'admittedly', 'it is true that', 'nonetheless'},
    'example': {'for example', 'for instance', 'such as', 'like', 'to illustrate', 'namely', 'including', 'in particular'},
    'purpose': {'so that', 'in order to', 'in order that', 'so as to', 'to', 'for the purpose of', 'with the aim of'},
    'time': {'first', 'then', 'next', 'after that', 'afterwards', 'before', 'finally', 'meanwhile',
             'eventually', 'at the same time', 'subsequently'},
    'summary': {'in conclusion', 'to conclude', 'in summary', 'to sum up', 'overall', 'in short', 'all in all', 'ultimately'}
}



#══════════════════════════════════════════════════════════════════════════════════════
#                           CHARGEMENT DES RESSOURCES NLP
#══════════════════════════════════════════════════════════════════════════════════════

def load_nlp_ressources():
    """
    Charge toutes les ressources NLP nécessaires (Spacy, NLTK, BERT).
    À appeler UNE SEULE FOIS au démarrage de l'API.
    """

    global nlp, stop_en, tokenizer

    try:
        #Télécharger les stopwords si nécessaire
        try:
            stopwords.words(STOPWORDS_LANGUAGE)
        except LookupError:
            nltk.download('stopwords', quiet=True)

        #Charger les stopwords
        stop_en = set(stopwords.words(STOPWORDS_LANGUAGE))

        #Charger Spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

        #Charger le tokenizer BERT
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

        print("✅ Ressources NLP chargées avec succès")
        return True
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement des ressources NLP : {e}")
        raise RuntimeError(f"Impossible de charger les ressources NLP : {e}")



#══════════════════════════════════════════════════════════════════════════════════════
#                        FONCTION DE PREPROCESSING / EXTRACTION
#══════════════════════════════════════════════════════════════════════════════════════

def is_word_tok(tok):
    """
    Vérifie si un token est un 'vrai mort' (alpha, nombre, URL)
    """
    return tok.is_alpha or tok.like_num or tok.like_url


def punctuation_features(text, len_words_value):
    """
    Extrait les features de ponctuation.
    """
    text = text or ""
    res = {}

    #Ellipses
    ellipses = text.count(ELLIPSIS_TOKEN)
    text_wo_ell = text.replace(ELLIPSIS_TOKEN, "")
    denominator = max(len_words_value, 1)
    res['punct_ellipsis_ratio'] = ellipses / denominator

    #Autres ponctuations
    for p in PUNCT_LIST:
        clean_p = re.sub(r'\W', '_', p)
        key = f"punct_{clean_p}_ratio"
        count = text_wo_ell.count(p)
        res[key] = count / denominator

    return res


def detect_connectives(text, len_word_value):
    """
    Détecte les connecteurs logiques.
    """
    text_lower = text.lower()
    detected = defaultdict(int)

    for category, phrases in connectives.items():
        for phrase in phrases:
            if phrase in text_lower:
                detected[category] += 1

    #Convertir en ratio
    denominator = max(len_word_value, 1)
    
    return {
        f"connective_{cat}_ratio": count / denominator for cat, count in detected.items()
    }


def pos_features(doc, len_words_value):
    """
    Extrait les features POS (Part-Of-Speech).
    """
    pos_counts = Counter(tok.pos_ for tok in doc if not tok.is_space)
    denominator = max(len_words_value, 1)

    return {
        f"pos_{pos}_ratio": pos_counts.get(pos, 0) / denominator for pos in ALL_POS_TAG
    }


def preprocessor(text):
    """
    Prétraite le texte et extrait les features nécessaires au modèle.

    Args:
        text (str): Texte brut à analyser

    Returns:
        pd.DataFrame: Ligne unique contenant toutes les features extraites
    """
    if not text or not isinstance(text, str):
        raise ValueError("Le texte doit être une chaîne de caractère non vide.")
    
    features = {}

    #Traitement avec Spacy
    doc = nlp(text)
    tokens = list(doc)

    #Mots valides
    valid_words = [tok for tok in tokens if is_word_tok(tok)]
    len_words = max(len(valid_words), 1) #Eviter les divisions par 0

    # === 1. Features de longueur ===
    features['len_chars'] = len(text)
    features['len_tokens_all'] = len(tokens)
    features['len_words'] = len_words

    #Phrases
    sentences = list(doc.sents)
    n_sentences = max(len(sentences), 1)
    features['n_sentences'] = n_sentences
    features['average_sentences_length'] = sum(len(
        [t for t in sent if not t.is_space ]) for sent in sentences) / n_sentences
    
    #Ratios
    features['len_chars_per_word'] = features['len_chars'] / len_words
    features['len_tokens_per_word'] = features['len_tokens_all'] /len_words

    # === 2. Ratios majuscules ===
    nb_uppercase = sum(1 for c in text if c.isupper())
    features['freq_uppercase'] = nb_uppercase / max(len(text), 1)

    # === 3. Features de ponctuation ===
    punct_features = punctuation_features(text, len_words)
    features.update(punct_features)

    # === 4. Ratio stopwords ===
    stopword_count = sum(1 for tok in valid_words if tok.text.lower() in stop_en)
    features['stopwords_ratio'] = stopword_count / len_words

    # === 5. Connecteurs logiques ===
    connective_features = detect_connectives(text, len_words)
    #Ajouter toutes les catégories (même si 0)
    for cat in connectives.keys():
        key = f"connective_{cat}_ratio"
        features['key'] = connective_features.get(key, 0.0)

    # === 6. POS tags ===
    pos_feat = pos_features(doc, len_words)
    features.update(pos_feat)

    #Convertir en DataFrame (une seule ligne)
    return pd.DataFrame([features])



#══════════════════════════════════════════════════════════════════════════════════════
#                        CHARGEMENT DU MODELE DE PREDICTION
#══════════════════════════════════════════════════════════════════════════════════════

def load_ml_model():
    """
    Charge le modèle de ML depuis MLflow.
    À appeler UNE SEULE FOIS au démarrage de l'API.
    """

    global model

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.sklearn.load_model(MODEL_URI)

        print("✅ Modèle ML chargé avec succès")
        return True
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        raise RuntimeError(f"Impossible de charger le modèle : {e}")



#══════════════════════════════════════════════════════════════════════════════════════
#                                   ENDPOINTS
#══════════════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    """
    Point d'entrée de l'API.
    
    Retourne les informations de base sur le service.
    """
    return "Retrouvez toute la documentation de l'API sur /docs"


@app.on_event("startup")
async def startup_event():
    """
    Chargement des ressources au démarrage de l'API.
    """
    print("🚀 Démarrage de l'API...")

    try:
        #Charger les ressources NLP
        load_nlp_ressources()

        #Charger le modèle
        load_ml_model()

        print("✅ API prête à recevoir des requêtes")

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE au démarrage : {e}")
        raise


@app.post("/predict", response_model=PredictionResponse, tags=["Détection"])
async def predict_single_text(input_data: TextInput):
    """
    Analyse un seul texte pour déterminé s'il a été généré par une IA.

    **Format accepté pour la requête :**
    json:
    `{
        "text": "Ici se trouve le texte."
    }`

    **Format accepté pour le champ "text" :**
    str `"Voici le texte."`

    **Retourne :**
    - `is_ai_generated` : 1 si le texte est généré par IA, 0 s'il est écrit par un humain

    - `message` : Message explicatif en langage naturel
    """

    #Preprocessing
    features_df = preprocessor(input_data.text)

    #Prédiction
    prediction = int(model.predict(features_df)[0])

    #Message explicatif
    if prediction == 1:
        message = "Cet avis semble avoir été généré par une intelligence artificielle."
    else:
        message = "Ce texte semble avoir été écrit par un humain."

    return PredictionResponse(
        is_ai_generated=prediction,
        message=message
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Détection"])
async def predict_batch_texts(input_data: BatchTextInput):
    """
    Analyse plusieurs textes simultanément pour déterminer s'ils ont été générés par une IA.

    **Limite :** Maximum 10,000 textes par requête

    **Format de la requête:**
    json:
    `{
        "texts": ["text1", "text2", ...]
    }`

    **Formats acceptés pour le champ "texts" :**
    - Liste de strings: `["text1", "text2", ...]`

    - Liste de dictionnaires: `[{"text": "text1}, {"texte": "text2}, ...]`

    **Retourne :**
    `predictions` : Liste des prédictions (1 = IA, 0 = Humain) dans le même ordre que les textes soumis
    """

    #Extraction des textes selon le format
    texts = []
    for item in input_data.texts:
        #Si c'est une liste de textes
        if isinstance(item, str):
            texts.append(item)
        #Sinon, si c'est une liste de dictionnaire
        elif isinstance(item, dict):
            #Et si ce dictionnaire contient une clé "text"
            if "text" in item:
                texts.append(item["text"])
            #Ou sinon si ce dictionnaire contient une clé "texte"
            elif "texte" in item:
                texts.append(item['texte'])
            #Sinon, on lève une erreur
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Les dictionnaires doivent contenir une clé 'texte."
                )
        #Si ce n'est ni une liste de texte ni une liste de dictionnaire, on lève une erreur
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Format non supporté: {type(item)}. Utilisez des strings ou des dictionnaires."
            )

    #Preprocessing
    features_list = pd.concat([preprocessor(text) for text in texts], ignore_index=True)

    #Predictions
    predictions = model.predict(features_list)

    #Conversion en liste
    predictions = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

    return BatchPredictionResponse(predictions=predictions)