import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import os

#══════════════════════════════════════════════════════════════════════════════════════
#════════════════════════════ CONFIGURATION ═══════════════════════════════════════════
#══════════════════════════════════════════════════════════════════════════════════════
#Configurer l'URI
mlflow.set_tracking_uri("https://julienrouillard-getaround-prediction.hf.space")

# Configuration AWS et MLflow
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://s3.eu-west-3.amazonaws.com"

ARTIFACT_STORE_URI = os.getenv("ARTIFACT_STORE_URI")



#══════════════════════════════════════════════════════════════════════════════════════
#════════════════════════════════ PREPARATION DES DONNES ══════════════════════════════
#══════════════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("../datasets_source/features_ia_vs_humain_ml.csv")
for col in ["classe", "SYM"]:
    if col in df.columns:
        df = df.drop(columns=[col])

target = "label"
features = [col for col in df.columns if col != target]

X = df[features]
Y = df[target]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

#Preprocessing
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#══════════════════════════════════════════════════════════════════════════════════════
#════════════════════════════ CREATION DU VOTING CLASSIFIER ═══════════════════════════ 
#══════════════════════════════════════════════════════════════════════════════════════
lr_best = LogisticRegression(
    C=45,
    max_iter=5000,
    penalty='l2',
    solver='lbfgs',
    random_state=42
)

rf_best = RandomForestClassifier(
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    n_estimators=300,
    random_state=42
)

xgb_best = XGBClassifier(
    gamma=0,
    learning_rate=0.3,
    max_depth=5,
    min_child_weight=1,
    n_estimators=300,
    subsample=1.0,
    random_state=42,
    eval_metric='logloss'
)

voting_best = VotingClassifier(
    estimators=[
        ('lr', lr_best),
        ('rf', rf_best),
        ('xgb', xgb_best)
    ],
    voting='soft',
    weights=[1, 2, 2]
)

#══════════════════════════════════════════════════════════════════════════════════════
#════════════════════════════════ ENTRAINEMENT ════════════════════════════════════════
#══════════════════════════════════════════════════════════════════════════════════════
voting_best.fit(X_train, Y_train)

#Prédictions
y_train_pred = voting_best.predict(X_train)
y_test_pred = voting_best.predict(X_test)

#Métriques
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_acc = accuracy_score(Y_train, y_train_pred)
test_acc = accuracy_score(Y_test, y_test_pred)
test_prec = precision_score(Y_test, y_test_pred)
test_rec = recall_score(Y_test, y_test_pred)
test_f1 = f1_score(Y_test, y_test_pred)

#══════════════════════════════════════════════════════════════════════════════════════
#═══════════════════════ ENREGISTREMENT SUR MLFLOW ════════════════════════════════════
#══════════════════════════════════════════════════════════════════════════════════════
mlflow.set_experiment("AI")

with mlflow.start_run(run_name="VotingClassifier_Best"):

    #Log des métriques
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_prec)
    mlflow.log_metric("test_recall", test_rec)
    mlflow.log_metric("test_f1_score", test_f1)
    
    # Log du modèle
    mlflow.sklearn.log_model(
        voting_best,
        artifact_path="voting_classifier",
        input_example=X_train[:5]
    )
    
    print("✅ Modèle enregistré sur MLflow !")
    print(f"📊 Test Accuracy: {test_acc:.3f}")
    print(f"📊 Test F1-Score: {test_f1:.3f}")