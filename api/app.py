#===========================================
#========== IMPORT DES LIBRAIRIES ==========
#===========================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union



#===================================================
#========== DESCRIPTION COMPLETE DE L'API ==========
#===================================================

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



#=========================================
# ========== MODELES DE DONNEES ==========
#=========================================

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



#===============================================
#========== FONCTION DE PREPROCESSING ==========
#===============================================

def preprocessor(text):
    """
    Prétraite le texte et extrait les features nécessaires au modèle.
    """
    features = "Ecrire le preprocessing"
    return features



#====================================================
#========== IMPORT DU MODELE DE PREDICTION ==========
#====================================================

model = "IMPORTER LE MODELE DEPUIS MLFLOW"



#=========================================
#=============== ENDPOINTS ===============
#=========================================

@app.get("/", tags=["Root"])
async def root():
    """
    Point d'entrée de l'API.
    
    Retourne les informations de base sur le service.
    """
    return "Retrouvez toute la documentation de l'API sur /docs"


@app.post("/predict", response_model=PredictionResponse, tags=["Détection"])
async def predict_single_text(input_data: TextInput):
    """
    Analyse un seul texte pour déterminé s'il a été généré par une IA.

    **Retourne :**
    - `is_ai_generated` : 1 si le texte est généré par IA, 0 s'il est écrit par un humain
    - `message` : Message explicatif en langage naturel
    """

    #Preprocessing
    features = preprocessor(input_data.text)

    #Prédiction
    prediction = model.predict(features)

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

    **Formats acceptés :**
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
                statsu_code=400,
                detail=f"Format non supporté: {type(item)}. Utilisez des strings ou des dictionnaires."
            )

    #Preprocessing
    features_list = [preprocessor(text) for text in texts]

    #Predictions
    predictions = model.predict(features_list)

    #Conversion en liste
    predictions = predictions.tolist() if hasattr(predictions, 'tolist  ') else list(predictions)

    return BatchPredictionResponse(predictions=predictions)