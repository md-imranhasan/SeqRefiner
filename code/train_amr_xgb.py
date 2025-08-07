import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib

# ----------------------------
# Load embeddings
# ----------------------------
df = pd.read_csv("amr_embeddings.csv")

# ----------------------------
# Remove classes with < 2 samples
# ----------------------------
class_counts = df["amr_class"].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()

if rare_classes:
    print(f"⚠️ Removing {len(rare_classes)} rare classes with < 2 samples:")
    print(rare_classes)
    df = df[~df["amr_class"].isin(rare_classes)]

# ----------------------------
# Features and Labels
# ----------------------------
X = df[[col for col in df.columns if col.startswith("f")]].values
y = df["amr_class"]

# ----------------------------
# Encode Labels
# ----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
)

# ----------------------------
# Train XGBoost
# ----------------------------
print("🚀 Training XGBoost classifier...")
clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_names),
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
clf.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = clf.predict(X_test)
print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# Optional: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# ----------------------------
# Save Model and Encoder
# ----------------------------
joblib.dump(clf, "xgb_amr_classifier.pkl")
joblib.dump(label_encoder, "amr_label_encoder.pkl")
print("✅ Model and label encoder saved.")
