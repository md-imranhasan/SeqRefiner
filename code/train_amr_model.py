import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Load and Clean Data
# ----------------------------
df = pd.read_csv("amr_embeddings.csv").dropna()

# Drop rare classes (keep only those with ≥10 samples)
class_counts = df["label"].value_counts()
valid_classes = class_counts[class_counts >= 10].index
df = df[df["label"].isin(valid_classes)]

print(f"✅ Using {len(df)} sequences across {len(valid_classes)} valid AMR classes")

# ----------------------------
# Prepare Features and Labels
# ----------------------------
X = df[[c for c in df.columns if c.startswith("f")]]
y = df["label"]

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------
# Compute Class Weights (✅ FIXED)
# ----------------------------
classes = np.array(sorted(y_train.unique()))  # Convert to NumPy array
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))

# ----------------------------
# Train Random Forest Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Visualize Confusion Matrix
# ----------------------------
print("📊 Saving confusion matrix as 'amr_confusion_matrix.png'")
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("amr_confusion_matrix.png", dpi=300)
plt.show()

# ----------------------------
# Save Trained Model
# ----------------------------
joblib.dump(model, "amr_model.pkl")
print("✅ Model saved as amr_model.pkl")
