
# train.py - trains multiple models and saves them
import pandas as pd, numpy as np, joblib, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

ROOT = Path(os.environ.get('PROJECT_ROOT', '/content/drive/MyDrive/MLOPS_Project'))
DATA_PATH = ROOT / 'uci_malware_detection.csv'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
# Expect Label column in first column; adapt if different
if 'Label' in df.columns:
    y = df['Label']
    X = df.drop(columns=['Label'])
else:
    y = df.iloc[:,0]
    X = df.iloc[:,1:]

# encode labels if needed
if X.shape[0] == 0:
    raise ValueError("No rows in data")
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)
    joblib.dump(le, MODELS_DIR / 'label_encoder.joblib')

# ensure numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5),
    "SGD_hinge": SGDClassifier(loss='hinge', max_iter=2000, tol=1e-3, random_state=42),
    "SGD_log": SGDClassifier(loss='log_loss', max_iter=2000, tol=1e-3, random_state=42),
    "GaussianNB": GaussianNB()
}

results = []
for name, clf in models.items():
    print("Training", name)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(name, "acc", acc)
    joblib.dump(clf, MODELS_DIR / f"{name}_model.pkl")
    results.append({'model':name, 'accuracy':acc})

# save comparison
import json
(pd.DataFrame(results).to_csv(MODELS_DIR / 'models_comparison.csv', index=False))
print("Saved models & comparison to", MODELS_DIR)
