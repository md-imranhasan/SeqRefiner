import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

# ----------------------------
# Load and Clean Data
# ----------------------------
df = pd.read_csv("amr_embeddings.csv").dropna()

# Keep only labels with ≥10 examples
label_counts = df["label"].value_counts()
valid_labels = label_counts[label_counts >= 10].index
df = df[df["label"].isin(valid_labels)]

X = df[[c for c in df.columns if c.startswith("f")]].values.astype(np.float32)
y_raw = df["label"].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ----------------------------
# Define MLP Model
# ----------------------------
class AMRMLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, output_dim=0):
        super().__init__()
        if output_dim == 0:
            raise ValueError("You must provide output_dim (number of classes)")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, X):
        return self.net(X)

# ----------------------------
# Train with Skorch
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = len(np.unique(y))

net = NeuralNetClassifier(
    AMRMLP,
    module__input_dim=X.shape[1],
    module__output_dim=n_classes,
    max_epochs=100,
    lr=1e-3,
    batch_size=64,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device=device,
)

print("🔁 Training MLP...")
net.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = net.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ----------------------------
# Confusion Matrix
# ----------------------------
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_encoder.classes_, xticks_rotation="vertical")
plt.title("MLP AMR Classifier — Confusion Matrix")
plt.tight_layout()
plt.savefig("mlp_confusion_matrix.png", dpi=300)
plt.show()

# ----------------------------
# Save Model and Label Encoder
# ----------------------------
import joblib
joblib.dump(net, "amr_mlp_model.pkl")
joblib.dump(label_encoder, "amr_label_encoder.pkl")
print("✅ MLP model saved as amr_mlp_model.pkl")
print("✅ Label encoder saved as amr_label_encoder.pkl")
