"""
Penguin Species Classifier using a Feedforward Neural Network
AI 100 - Midterm Project
Authors: Josh Varughese, Dhruti Thakur

This script trains a deep learning model (feedforward neural network)
to classify penguin species using the Palmer Penguins dataset.

Requirements:
    pip install torch pandas numpy scikit-learn matplotlib

Usage:
    python penguin_classifier.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── 1. Load & Clean Data ────────────────────────────────────────────────────

URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
df = pd.read_csv(URL)

# Drop rows with missing values
df = df.dropna(subset=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species'])

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Species distribution:\n{df['species'].value_counts()}\n")

# ─── 2. Prepare Features & Labels ────────────────────────────────────────────

FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = df[FEATURES].values
le = LabelEncoder()
y = le.fit_transform(df['species'])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# ─── 3. Define the Neural Network ────────────────────────────────────────────

class PenguinNet(nn.Module):
    def __init__(self, input_dim=4, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model     = PenguinNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ─── 4. Train ────────────────────────────────────────────────────────────────

EPOCHS = 100
train_losses = []

print("Training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    avg_loss = epoch_loss / len(X_train)
    train_losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ─── 5. Evaluate ─────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    y_pred = logits.argmax(dim=1).numpy()

acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─── 6. Plot Results ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(train_losses, color='steelblue')
axes[0].set_title("Training Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].grid(True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks(range(3)); axes[1].set_yticks(range(3))
axes[1].set_xticklabels(le.classes_, rotation=15)
axes[1].set_yticklabels(le.classes_)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].set_title("Confusion Matrix")
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()
print("\nPlot saved to results.png")
