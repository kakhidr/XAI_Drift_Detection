import os
import random
import numpy as np
import torch

SEED = 42
# ===============================
# Attack Hyperparameters
# ===============================
FGSM_EPS = 0.001
PGD_EPS = 0.001
PGD_ALPHA = 0.01
PGD_ITERS = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Seed locked:", SEED)
print("Torch:", torch.__version__)
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

# Tiny toy model
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)

model = ToyModel()
model.eval()

# Fake input
x = torch.tensor([[0.2, 0.5, 0.3]], requires_grad=True)

# Integrated Gradients
ig = IntegratedGradients(model)
attr = ig.attribute(x, target=0)

print("Attributions:", attr)

## Data loading
from pathlib import Path
import pandas as pd

DATASET_FOLDERS = {
    "CIC_IDS_2018": "/workspaces/XAI_Drift_Detection/CIS/CSE-CIC-IDS2018",
    "BETH": "/workspaces/XAI_Drift_Detection/configs/Beth" 
}

# =========================
# Folder selection
# =========================
print("\nAvailable dataset folders:")
FOLDER_LIST = list(DATASET_FOLDERS.keys())
for i, name in enumerate(FOLDER_LIST, 1):
    print(f"{i}. {name}")

folder_choice = input("\nEnter folder number: ").strip()
if not folder_choice.isdigit() or int(folder_choice) < 1 or int(folder_choice) > len(FOLDER_LIST):
    raise ValueError("Invalid folder number.")

selected_folder = FOLDER_LIST[int(folder_choice) - 1]
DATA_DIR = Path(DATASET_FOLDERS[selected_folder])

print(f"\nSelected folder: {selected_folder}")
print("Path exists:", DATA_DIR.exists())

# --- Quick label inspection (handles both CIC and new dataset) ---
print("\nInspecting label distributions for first 50,000 rows of each CSV:")

for csv in sorted(DATA_DIR.glob("*.csv")):
    try:
        df_tmp = pd.read_csv(csv, nrows=50000)

        print(f"{csv.name}:")

        if "Label" in df_tmp.columns:
            # CIC dataset
            counts = df_tmp["Label"].value_counts()
            print("Label distribution:")
            print(counts.head())

        elif "sus" in df_tmp.columns and "evil" in df_tmp.columns:
            # Beth dataset (combined label)
            combined = ((df_tmp["sus"] == 1) | (df_tmp["evil"] == 1)).astype(int)
            print("Combined (sus OR evil) distribution:")
            print(combined.value_counts())

            print("\nIndividual distributions:")
            print("sus:")
            print(df_tmp["sus"].value_counts())
            print("evil:")
            print(df_tmp["evil"].value_counts())

        elif "evil" in df_tmp.columns:
            print("evil distribution:")
            print(df_tmp["evil"].value_counts())

        elif "sus" in df_tmp.columns:
            print("sus distribution:")
            print(df_tmp["sus"].value_counts())

        else:
            print("No known label columns found.")

        print("-" * 40)

    except Exception as e:
        print(f"{csv.name}: Error reading file -> {e}")
        print("-" * 40)
# =========================
# Automatically detect CSVs in the selected folder
# =========================
csv_files = sorted(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise ValueError("No CSV files found in selected folder.")

DATASET_MAP = {f.stem: f.name for f in csv_files}

# =========================
# Dataset selection
# =========================
print("\nAvailable datasets:")
DATASET_LIST = list(DATASET_MAP.keys())
for i, name in enumerate(DATASET_LIST, 1):
    print(f"{i}. {name} --> {DATASET_MAP[name]}")

dataset_choice = input("\nEnter dataset number: ").strip()
if not dataset_choice.isdigit() or int(dataset_choice) < 1 or int(dataset_choice) > len(DATASET_LIST):
    raise ValueError("Invalid dataset number.")

user_choice = DATASET_LIST[int(dataset_choice) - 1]
csv_file = DATA_DIR / DATASET_MAP[user_choice]

print(f"\nYou selected: {user_choice} ({csv_file.name})")

# =========================
# Attack selection (FGSM / PGD)
# =========================
print("\nAttack options:")
print("1. FGSM")
print("2. PGD")
print("3. Both FGSM + PGD")

attack_choice = input("Enter choice (1/2/3): ").strip()

RUN_FGSM = False
RUN_PGD = False

if attack_choice == "1":
    RUN_FGSM = True
elif attack_choice == "2":
    RUN_PGD = True
elif attack_choice == "3":
    RUN_FGSM = True
    RUN_PGD = True
else:
    raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

print(f"FGSM enabled: {RUN_FGSM}")
print(f"PGD enabled: {RUN_PGD}")
# =========================
# Load dataset
# =========================
df = pd.read_csv(csv_file) # load entire file for full analysis, adjust nrows if needed for quick testing

print("Loaded:", csv_file.name) # show which file was loaded
print("Shape:", df.shape) # show number of rows and columns to check data loaded correctly

# =========================
# Safe label column detection
# =========================
if "Label" in df.columns:
    LABEL_COL = "Label"
elif "sus" in df.columns and "evil" in df.columns:
    LABEL_COL = ["sus", "evil"]
elif "evil" in df.columns:
    LABEL_COL = "evil"
elif "sus" in df.columns:
    LABEL_COL = "sus"
else:
    LABEL_COL = None

# =========================
# Original checks (kept as-is but safe)
# =========================
if "Label" in df.columns:
    print(df["Label"].value_counts())
else:
    print("Label column not found for direct inspection")

df.head(3)

df.columns.tolist()[:30], df.columns.tolist()[-10:]

[label for label in df.columns if "label" in label.lower()]

# =========================
# Label processing (UPDATED but same structure)
# =========================
if LABEL_COL == "Label":
    print(df[LABEL_COL].value_counts().head(20))
    y = (df[LABEL_COL].astype(str).str.lower() != "benign").astype(int)

elif LABEL_COL == ["sus", "evil"]:
    print("\nsus distribution:")
    print(df["sus"].value_counts())

    print("\nevil distribution:")
    print(df["evil"].value_counts())

    y = ((df["sus"] == 1) | (df["evil"] == 1)).astype(int)

elif LABEL_COL == "evil":
    print(df["evil"].value_counts())
    y = df["evil"].astype(int)

elif LABEL_COL == "sus":
    print(df["sus"].value_counts())
    y = df["sus"].astype(int)

else:
    raise ValueError("No valid label column found")

print("Binary label counts (0=benign, 1=attack):")
print(y.value_counts())

# =========================
# Feature processing 
# =========================
X = df.drop(columns=[LABEL_COL], errors="ignore") if isinstance(LABEL_COL, str) else df.drop(columns=LABEL_COL, errors="ignore")

X_num = X.select_dtypes(include="number")

print("Total columns:", X.shape[1])
print("Numeric columns:", X_num.shape[1])
print("Any NaNs:", X_num.isna().any().any())

X_num.head(3)

# =========================
# SECOND LABEL BLOCK (kept EXACT structure, just fixed)
# =========================
if LABEL_COL == "Label":
    LABEL_COL = "Label"
    y = (df[LABEL_COL].astype(str).str.lower() != "benign").astype(int)

elif LABEL_COL == ["sus", "evil"]:
    LABEL_COL = ["sus", "evil"]
    y = ((df["sus"] == 1) | (df["evil"] == 1)).astype(int)

elif LABEL_COL == "evil":
    LABEL_COL = "evil"
    y = df["evil"].astype(int)

elif LABEL_COL == "sus":
    LABEL_COL = "sus"
    y = df["sus"].astype(int)

print("Binary label distribution:")
print(y.value_counts())

# =========================
# FINAL FEATURE MATRIX (same as your code)
# =========================
X = df.drop(columns=[LABEL_COL], errors="ignore") if isinstance(LABEL_COL, str) else df.drop(columns=LABEL_COL, errors="ignore")

X = X.select_dtypes(include="number")

print("Feature matrix shape:", X.shape)
print("Any NaNs:", X.isna().any().any())
print("Any infs:", (~np.isfinite(X.to_numpy())).any())
# Train-test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Train label distribution:")
print(y_train.value_counts())

# Feature scaling (standardization)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create validation split from training data

from sklearn.model_selection import train_test_split

X_train2, X_val, y_train2, y_val = train_test_split(
    X_train_scaled,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Scaled train mean (first 5 features):", X_train_scaled.mean(axis=0)[:5])
print("Scaled train std (first 5 features):", X_train_scaled.std(axis=0)[:5])

# Convert to PyTorch tensors

import torch

X_train_t = torch.tensor(X_train2, dtype=torch.float32)
y_train_t = torch.tensor(y_train2.values, dtype=torch.long)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.long)

X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

print("Train:", X_train_t.shape)
print("Validation:", X_val_t.shape)
print("Test:", X_test_t.shape)   

print(X_train_t.shape, y_train_t.shape) # show shapes of training tensors to confirm they look correct    
print(X_test_t.shape, y_test_t.shape) # show shapes of test tensors to confirm they look correct

# Define a simple MLP model for IDS

import torch.nn as nn # already imported above, but re-importing here for clarity in this cell

class IDS_MLP(nn.Module): # simple MLP for binary classification
    def __init__(self, input_dim): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x): # forward pass through the network
        return self.net(x)

model = IDS_MLP(input_dim=X_train_t.shape[1]) # initialize
model

# Training setup

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# Create DataLoader for training
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop

model.train()

EPOCHS = 20
patience = 5
best_val_loss = float("inf")
early_stop_counter = 0

for epoch in range(EPOCHS):

    # Training
    model.train()
    train_loss = 0

    for xb, yb in train_loader:

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            logits = model(xb)

            loss = criterion(logits, yb)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # Early stopping
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        early_stop_counter = 0

        torch.save(model.state_dict(), "best_model.pt")

    else:

        early_stop_counter += 1

        if early_stop_counter >= patience:

            print("Early stopping triggered")
            break
        
    # Evaluation on test set

model.eval() # set model to evaluation mode
with torch.no_grad(): # no need to track gradients during evaluation
    logits = model(X_test_t)
    preds = torch.argmax(logits, dim=1)

from sklearn.metrics import classification_report # already imported above, but re-importing here for clarity in this cell

print(classification_report(y_test_t.numpy(), preds.numpy(), digits=4)) # show precision, recall, f1-score for each class, and overall accuracy

# Check prediction distribution to see if model is predicting both classes or just one (common issue with imbalanced data)

model.eval() # set model to evaluation mode

with torch.no_grad(): # no need to track gradients during evaluation
    logits = model(X_test_t) # get raw output scores (logits) from the model for the test set
    preds = torch.argmax(logits, dim=1) # get predicted class labels by taking the index of the max logit for each sample

print("Prediction distribution:")
print(torch.bincount(preds))

# Check how many samples were classified correctly to get a sense of model performance beyond just the classification report

correct_mask = preds == y_test_t # create a boolean mask where True indicates a correct prediction and False indicates an incorrect prediction
num_correct = correct_mask.sum().item() # sum the True values in the mask to get the total number of correct predictions, convert to Python int with .item()

print(f"Correctly classified samples: {num_correct} / {len(y_test_t)}") # show number of correctly classified samples out of total test samples to get a sense of overall performance

# Select a subset of correctly classified samples for evaluation of explanations, ensuring we have a fixed budget for evaluation

MAX_EVAL = 500

correct_indices = torch.where(correct_mask)[0]

correct_labels = y_test_t[correct_indices]

benign_idx = correct_indices[correct_labels == 0]
attack_idx = correct_indices[correct_labels == 1]

n_each = MAX_EVAL // 2

eval_indices = torch.cat([
    benign_idx[:n_each],
    attack_idx[:n_each]
])

X_eval = X_test_t[eval_indices] # select the feature tensors for the evaluation samples using the eval_indices to index into the test feature tensor
y_eval = y_test_t[eval_indices] # select the true labels for the evaluation samples using the eval_indices to index into the test label tensor
preds_eval = preds[eval_indices] # select the predicted labels for the evaluation samples using the eval_indices to index into the predicted labels tensor

print("Eval feature tensor shape:", X_eval.shape)
print("Eval label distribution:", torch.bincount(y_eval))

# Sanity check: all eval samples are prediction-consistent
assert torch.all(preds_eval == y_eval), "Eval set contains misclassified samples!"

print("Evaluation set frozen — all predictions correct.")

# ===============================
# 8. FGSM Adversarial Attack
# ===============================
import torch.nn.functional as F

def fgsm_attack(model, X, y, epsilon):
    """
    Fast Gradient Sign Method attack
    X: input tensor
    y: true labels (tensor)
    epsilon: perturbation size
    """
    X_adv = X.clone().detach().requires_grad_(True)
    model.eval()
    outputs = model(X_adv)
    loss = F.cross_entropy(outputs, y)
    model.zero_grad()
    loss.backward()
    X_adv = X_adv + epsilon * X_adv.grad.sign()
    return X_adv.detach()

# Generate FGSM adversarial examples for evaluation set
X_adv_fgsm = None
if RUN_FGSM:
    X_adv_fgsm = fgsm_attack(model, X_eval, y_eval, epsilon= FGSM_EPS)
    print("FGSM attack generated")

# ===============================
# 9. PGD Adversarial Attack
# ===============================
def pgd_attack(model, X, y, epsilon, alpha, iters):
    """
    Projected Gradient Descent attack
    X: input tensor
    y: true labels (tensor)
    epsilon: max perturbation
    alpha: step size
    iters: number of iterations
    """
    X_adv = X.clone().detach()
    X_orig = X.clone().detach()
    for i in range(iters):
        X_adv.requires_grad_(True)
        outputs = model(X_adv)
        loss = F.cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()
        X_adv = X_adv + alpha * X_adv.grad.sign()
        # Project back to epsilon-ball around original input
        X_adv = torch.min(torch.max(X_adv, X_orig - epsilon), X_orig + epsilon).detach()
    return X_adv

# Generate PGD adversarial examples for evaluation set
X_adv_pgd = None
if RUN_PGD:
    X_adv_pgd = pgd_attack(model, X_eval, y_eval, epsilon=PGD_EPS, alpha=PGD_ALPHA, iters=PGD_ITERS)
    print("PGD attack generated")

# ===============================
# 10. Integrated Gradients (IG)
# ===============================
from captum.attr import IntegratedGradients

model.eval()
ig = IntegratedGradients(model)

def get_ig(x_tensor, target=1):
    attr = ig.attribute(x_tensor, target=target)
    return attr.detach().cpu().numpy()

# IG attributions for evaluation set (before adversarial attacks)
IG_clean_eval = get_ig(X_eval, target=1)

# ===============================
# 11. SHAP
# ===============================
import shap

# Use tensor directly (do NOT convert to NumPy for DeepExplainer)
explainer_shap = shap.DeepExplainer(model, X_train_t[:100])

def get_shap(x_tensor):
    s = explainer_shap.shap_values(x_tensor)
    if isinstance(s, list):
        s = s[1]  # class 1
    return np.array(s).squeeze()

# SHAP attributions for evaluation set (before adversarial attacks)
SHAP_clean_eval = get_shap(X_eval)

# ===============================
# 12. Adversarial prediction check and preserved samples + attack success
# ===============================
model.eval()

def preserved_and_attack_stats(X_clean, X_adv, y_true, name="Attack"):
    with torch.no_grad():
        pred_clean = torch.argmax(model(X_clean), dim=1)
        pred_adv   = torch.argmax(model(X_adv), dim=1)

    mask_preserved = pred_clean == pred_adv
    num_preserved = mask_preserved.sum().item()
    num_total = X_clean.shape[0]
    num_flipped = num_total - num_preserved
    attack_success_rate = num_flipped / num_total * 100

    print(f"{name} preserved samples: {num_preserved} / {num_total}")
    print(f"{name} flipped samples (attack success): {num_flipped} / {num_total} ({attack_success_rate:.2f}%)")

    return X_clean[mask_preserved], X_adv[mask_preserved], y_true[mask_preserved]

# FGSM
if X_adv_fgsm is not None:
    X_clean_fgsm_preserved, X_adv_fgsm_preserved, y_fgsm_preserved = preserved_and_attack_stats(
        X_eval, X_adv_fgsm, y_eval, name="FGSM"
    )

# PGD
if X_adv_pgd is not None:
    X_clean_pgd_preserved, X_adv_pgd_preserved, y_pgd_preserved = preserved_and_attack_stats(
        X_eval, X_adv_pgd, y_eval, name="PGD"
    )

# ===============================
# 13. Batch-friendly IG and SHAP functions
# ===============================
def compute_ig_attributions(model, X, target=1, batch_size=64):
    attributions = []
    model.eval()
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size].clone().requires_grad_(True)
        attr = ig.attribute(batch, target=target)
        attributions.append(attr.detach().cpu().numpy())
    return np.concatenate(attributions, axis=0)

def compute_shap_attributions(explainer, X, batch_size=64):
    attributions = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size]
        s = explainer.shap_values(batch)
        if isinstance(s, list):
            s = s[1]
        attributions.append(s)
    return np.concatenate(attributions, axis=0)

# ===============================
# 14. Compute attributions for preserved samples
# ===============================
# FGSM
if RUN_FGSM:
    IG_clean_fgsm = compute_ig_attributions(model, X_clean_fgsm_preserved, target=1)
    IG_adv_fgsm   = compute_ig_attributions(model, X_adv_fgsm_preserved, target=1)

    S_clean_fgsm  = compute_shap_attributions(explainer_shap, X_clean_fgsm_preserved)
    S_adv_fgsm    = compute_shap_attributions(explainer_shap, X_adv_fgsm_preserved)

# PGD
if RUN_PGD:
    IG_clean_pgd = compute_ig_attributions(model, X_clean_pgd_preserved, target=1)
    IG_adv_pgd   = compute_ig_attributions(model, X_adv_pgd_preserved, target=1)

    S_clean_pgd  = compute_shap_attributions(explainer_shap, X_clean_pgd_preserved)
    S_adv_pgd    = compute_shap_attributions(explainer_shap, X_adv_pgd_preserved)

# ===============================
# 15. Drift Metrics: Cosine and Euclidean Distances
# ===============================
from sklearn.metrics.pairwise import cosine_distances

def compute_distances(A_clean, A_adv):
    A_clean = A_clean.reshape(A_clean.shape[0], -1)
    A_adv   = A_adv.reshape(A_adv.shape[0], -1)

    cosine_d = np.array([cosine_distances(A_clean[i:i+1], A_adv[i:i+1])[0,0] for i in range(A_clean.shape[0])])
    euclid_d = np.linalg.norm(A_clean - A_adv, axis=1)
    return cosine_d, euclid_d

def print_drift_stats(name, distances):
    print(f"{name} drift stats:")
    print("Min:", distances.min())
    print("Mean:", distances.mean())
    print("Max:", distances.max())
    print("-"*40)

from scipy.special import softmax
from scipy.stats import entropy  # KL divergence

def compute_kl_divergence(A_clean, A_adv, eps=1e-10):
    """
    Compute KL divergence between clean and adversarial attributions.
    Converts attributions to probability distributions using softmax.
    """
    A_clean = A_clean.reshape(A_clean.shape[0], -1)
    A_adv   = A_adv.reshape(A_adv.shape[0], -1)

    kl_vals = []

    for i in range(A_clean.shape[0]):
        p = softmax(A_clean[i]) + eps
        q = softmax(A_adv[i]) + eps

        kl = entropy(p, q)  # KL(p || q)
        kl_vals.append(kl)

    return np.array(kl_vals)

# ===============================
# 16. Compute and display drift for all combinations
# ===============================
# ===============================
# 16. Compute and display drift
# ===============================

# ---------- IG + FGSM ----------
if RUN_FGSM:
    cos_ig_fgsm, euc_ig_fgsm = compute_distances(IG_clean_fgsm, IG_adv_fgsm)
    print_drift_stats("IG + FGSM Cosine", cos_ig_fgsm)
    print_drift_stats("IG + FGSM Euclidean", euc_ig_fgsm)

# ---------- IG + PGD ----------
if RUN_PGD:
    cos_ig_pgd, euc_ig_pgd = compute_distances(IG_clean_pgd, IG_adv_pgd)
    print_drift_stats("IG + PGD Cosine", cos_ig_pgd)
    print_drift_stats("IG + PGD Euclidean", euc_ig_pgd)

# ---------- SHAP + FGSM ----------
if RUN_FGSM:
    cos_shap_fgsm, euc_shap_fgsm = compute_distances(S_clean_fgsm, S_adv_fgsm)
    print_drift_stats("SHAP + FGSM Cosine", cos_shap_fgsm)
    print_drift_stats("SHAP + FGSM Euclidean", euc_shap_fgsm)

# ---------- SHAP + PGD ----------
if RUN_PGD:
    cos_shap_pgd, euc_shap_pgd = compute_distances(S_clean_pgd, S_adv_pgd)
    print_drift_stats("SHAP + PGD Cosine", cos_shap_pgd)
    print_drift_stats("SHAP + PGD Euclidean", euc_shap_pgd)

# ---------- FGSM ----------
if RUN_FGSM:
    S_clean_fgsm_flat = S_clean_fgsm.reshape(S_clean_fgsm.shape[0], -1)
    IG_clean_fgsm_flat = IG_clean_fgsm.reshape(IG_clean_fgsm.shape[0], -1)

    cos_check_ig_fgsm = np.array([
        cosine_distances(IG_clean_fgsm_flat[i:i+1], IG_clean_fgsm_flat[i:i+1])[0,0]
        for i in range(IG_clean_fgsm_flat.shape[0])
    ])

    cos_check_shap_fgsm = np.array([
        cosine_distances(S_clean_fgsm_flat[i:i+1], S_clean_fgsm_flat[i:i+1])[0,0]
        for i in range(S_clean_fgsm_flat.shape[0])
    ])

    print("FGSM Sanity Check (IG):",
          cos_check_ig_fgsm.min(), cos_check_ig_fgsm.mean(), cos_check_ig_fgsm.max())

    print("FGSM Sanity Check (SHAP):",
          cos_check_shap_fgsm.min(), cos_check_shap_fgsm.mean(), cos_check_shap_fgsm.max())
if RUN_FGSM:
    kl_ig_fgsm = compute_kl_divergence(IG_clean_fgsm, IG_adv_fgsm)
    print_drift_stats("IG + FGSM KL", kl_ig_fgsm)
if RUN_FGSM:
    kl_shap_fgsm = compute_kl_divergence(S_clean_fgsm, S_adv_fgsm)
    print_drift_stats("SHAP + FGSM KL", kl_shap_fgsm)

# ---------- PGD ----------
if RUN_PGD:
    S_clean_pgd_flat = S_clean_pgd.reshape(S_clean_pgd.shape[0], -1)
    IG_clean_pgd_flat = IG_clean_pgd.reshape(IG_clean_pgd.shape[0], -1)

    cos_check_ig_pgd = np.array([
        cosine_distances(IG_clean_pgd_flat[i:i+1], IG_clean_pgd_flat[i:i+1])[0,0]
        for i in range(IG_clean_pgd_flat.shape[0])
    ])

    cos_check_shap_pgd = np.array([
        cosine_distances(S_clean_pgd_flat[i:i+1], S_clean_pgd_flat[i:i+1])[0,0]
        for i in range(S_clean_pgd_flat.shape[0])
    ])

if RUN_PGD:
    kl_ig_pgd = compute_kl_divergence(IG_clean_pgd, IG_adv_pgd)
    print_drift_stats("IG + PGD KL", kl_ig_pgd)

if RUN_PGD:
    kl_shap_pgd = compute_kl_divergence(S_clean_pgd, S_adv_pgd)
    print_drift_stats("SHAP + PGD KL", kl_shap_pgd)

print("PGD Sanity Check (IG):",
          cos_check_ig_pgd.min(), cos_check_ig_pgd.mean(), cos_check_ig_pgd.max())

print("PGD Sanity Check (SHAP):",
          cos_check_shap_pgd.min(), cos_check_shap_pgd.mean(), cos_check_shap_pgd.max())
# ===============================
# 18. Drift Visualization: Cosine + Euclidean
# ===============================
import matplotlib.pyplot as plt
import os

# Create folder to save plots
PLOT_DIR = "drift_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_drift_histogram(distances, title, filename, epsilon, info, bins=30):
    plt.figure(figsize=(12,8))

    plt.hist(distances, bins=bins, edgecolor='black')

    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Frequency")

    # ✅ epsilon + info inside graph
    text_str = f"ε = {epsilon}\n{info}"

    plt.text(
        0.95, 0.95,
        text_str,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Saved: {filename}")

plot_drift_histogram(
    cos_ig_fgsm,
    "IG + FGSM Cosine Drift",
    os.path.join(PLOT_DIR, "ig_fgsm_cosine.png"),
    FGSM_EPS,
    "Cosine: direction change in explanations"
)

plot_drift_histogram(
    euc_ig_fgsm,
    "IG + FGSM Euclidean Drift",
    os.path.join(PLOT_DIR, "ig_fgsm_euclidean.png"),
    FGSM_EPS,
    "Euclidean: magnitude change in explanations"
)

plot_drift_histogram(
    kl_ig_fgsm,
    "IG + FGSM KL Drift",
    os.path.join(PLOT_DIR, "ig_fgsm_kl.png"),
    FGSM_EPS,
    "KL: distribution shift in explanations"
)
plot_drift_histogram(
    cos_ig_pgd,
    "IG + PGD Cosine Drift",
    os.path.join(PLOT_DIR, "ig_pgd_cosine.png"),
    PGD_EPS,
    "Cosine: direction change in explanations"
)

plot_drift_histogram(
    euc_ig_pgd,
    "IG + PGD Euclidean Drift",
    os.path.join(PLOT_DIR, "ig_pgd_euclidean.png"),
    PGD_EPS,
    "Euclidean: magnitude change in explanations"
)

plot_drift_histogram(
    kl_ig_pgd,
    "IG + PGD KL Drift",
    os.path.join(PLOT_DIR, "ig_pgd_kl.png"),
    PGD_EPS,
    "KL: distribution shift in explanations"
)
plot_drift_histogram(
    cos_shap_fgsm,
    "SHAP + FGSM Cosine Drift",
    os.path.join(PLOT_DIR, "shap_fgsm_cosine.png"),
    FGSM_EPS,
    "Cosine: direction change in explanations"
)

plot_drift_histogram(
    euc_shap_fgsm,
    "SHAP + FGSM Euclidean Drift",
    os.path.join(PLOT_DIR, "shap_fgsm_euclidean.png"),
    FGSM_EPS,
    "Euclidean: magnitude change in explanations"
)

plot_drift_histogram(
    kl_shap_fgsm,
    "SHAP + FGSM KL Drift",
    os.path.join(PLOT_DIR, "shap_fgsm_kl.png"),
    FGSM_EPS,
    "KL: distribution shift in explanations"
)
plot_drift_histogram(
    cos_shap_pgd,
    "SHAP + PGD Cosine Drift",
    os.path.join(PLOT_DIR, "shap_pgd_cosine.png"),
    PGD_EPS,
    "Cosine: direction change in explanations"
)

plot_drift_histogram(
    euc_shap_pgd,
    "SHAP + PGD Euclidean Drift",
    os.path.join(PLOT_DIR, "shap_pgd_euclidean.png"),
    PGD_EPS,
    "Euclidean: magnitude change in explanations"
)

plot_drift_histogram(
    kl_shap_pgd,
    "SHAP + PGD KL Drift",
    os.path.join(PLOT_DIR, "shap_pgd_kl.png"),
    PGD_EPS,
    "KL: distribution shift in explanations"
)
    
# ===============================
# 20. Visualization of Drift and ROC for Codespaces
# ===============================
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Codespaces
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_histogram(distances, title, filename, bins=30):
    plt.figure(figsize=(12,8))
    plt.hist(distances, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved histogram: {filename}")

def plot_roc(y_true, drift_scores, title, filename):
    fpr, tpr, _ = roc_curve(y_true, drift_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(12,12))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved ROC plot: {filename}")

# ===============================
# Prepare labels + scores (with KL)
# ===============================
def prepare_labels_and_scores_with_kl(clean_attr, adv_attr, distances_cos, distances_euc, distances_kl):
    """
    Prepares labels and drift scores for ROC curves.

    clean_attr: attributions for clean samples
    adv_attr: attributions for adversarial samples
    distances_cos: cosine distances
    distances_euc: euclidean distances
    distances_kl: KL divergence values
    """

    # Labels: 0 = clean, 1 = adversarial
    y_labels = np.concatenate([
        np.zeros(len(clean_attr)),
        np.ones(len(adv_attr))
    ])

    # Scores
    scores_cos = np.concatenate([
        np.zeros(len(distances_cos)),
        distances_cos
    ])

    scores_euc = np.concatenate([
        np.zeros(len(distances_euc)),
        distances_euc
    ])

    scores_kl = np.concatenate([
        np.zeros(len(distances_kl)),
        distances_kl
    ])

    return y_labels, scores_cos, scores_euc, scores_kl
# -------------------------------
# ROC curves for drift detection
# -------------------------------
def prepare_labels_and_scores(clean_attr, adv_attr, distances_cos, distances_euc):
    # Labels: 0 = clean, 1 = adversarial
    y_labels = np.concatenate([np.zeros(len(clean_attr)), np.ones(len(adv_attr))])
    # Drift scores: distance of clean=0, adv=distance
    scores_cos = np.concatenate([np.zeros(len(distances_cos)), distances_cos])
    scores_euc = np.concatenate([np.zeros(len(distances_euc)), distances_euc])
    return y_labels, scores_cos, scores_euc

# IG + FGSM
if RUN_FGSM:
    y_ig_fgsm, ig_fgsm_scores_cos, ig_fgsm_scores_euc = prepare_labels_and_scores(
        IG_clean_fgsm, IG_adv_fgsm, cos_ig_fgsm, euc_ig_fgsm
    )
    plot_roc(y_ig_fgsm, ig_fgsm_scores_cos, "IG + FGSM Cosine Drift ROC", "roc_ig_fgsm_cos.png")
    plot_roc(y_ig_fgsm, ig_fgsm_scores_euc, "IG + FGSM Euclidean Drift ROC", "roc_ig_fgsm_euc.png")

# IG + PGD
if RUN_PGD:
    y_ig_pgd, ig_pgd_scores_cos, ig_pgd_scores_euc = prepare_labels_and_scores(
        IG_clean_pgd, IG_adv_pgd, cos_ig_pgd, euc_ig_pgd
    )
    plot_roc(y_ig_pgd, ig_pgd_scores_cos, "IG + PGD Cosine Drift ROC", "roc_ig_pgd_cos.png")
    plot_roc(y_ig_pgd, ig_pgd_scores_euc, "IG + PGD Euclidean Drift ROC", "roc_ig_pgd_euc.png")

# SHAP + FGSM
if RUN_FGSM:
    y_shap_fgsm, shap_fgsm_scores_cos, shap_fgsm_scores_euc = prepare_labels_and_scores(
        S_clean_fgsm, S_adv_fgsm, cos_shap_fgsm, euc_shap_fgsm
    )
    plot_roc(y_shap_fgsm, shap_fgsm_scores_cos, "SHAP + FGSM Cosine Drift ROC", "roc_shap_fgsm_cos.png")
    plot_roc(y_shap_fgsm, shap_fgsm_scores_euc, "SHAP + FGSM Euclidean Drift ROC", "roc_shap_fgsm_euc.png")

# SHAP + PGD
if RUN_PGD:
    y_shap_pgd, shap_pgd_scores_cos, shap_pgd_scores_euc = prepare_labels_and_scores(
        S_clean_pgd, S_adv_pgd, cos_shap_pgd, euc_shap_pgd
    )
    plot_roc(y_shap_pgd, shap_pgd_scores_cos, "SHAP + PGD Cosine Drift ROC", "roc_shap_pgd_cos.png")
    plot_roc(y_shap_pgd, shap_pgd_scores_euc, "SHAP + PGD Euclidean Drift ROC", "roc_shap_pgd_euc.png")

# ===============================
# IG + FGSM KL ROC
# ===============================
if RUN_FGSM:
    y_ig_fgsm, cos_ig_fgsm_scores, euc_ig_fgsm_scores, kl_ig_fgsm_scores = prepare_labels_and_scores_with_kl(
        IG_clean_fgsm,
        IG_adv_fgsm,
        cos_ig_fgsm,
        euc_ig_fgsm,
        kl_ig_fgsm
    )

    plot_roc(
        y_ig_fgsm,
        kl_ig_fgsm_scores,
        "IG + FGSM KL Drift ROC",
        "roc_ig_fgsm_kl.png"
    )
# ===============================
# IG + PGD KL ROC
# ===============================
if RUN_PGD:
    y_ig_pgd, cos_ig_pgd_scores, euc_ig_pgd_scores, kl_ig_pgd_scores = prepare_labels_and_scores_with_kl(
        IG_clean_pgd,
        IG_adv_pgd,
        cos_ig_pgd,
        euc_ig_pgd,
        kl_ig_pgd
    )

    plot_roc(
        y_ig_pgd,
        kl_ig_pgd_scores,
        "IG + PGD KL Drift ROC",
        "roc_ig_pgd_kl.png"
    )

# ===============================
# SHAP + PGD KL ROC
# ===============================
if RUN_PGD:
    y_shap_pgd, cos_shap_pgd_scores, euc_shap_pgd_scores, kl_shap_pgd_scores = prepare_labels_and_scores_with_kl(
        S_clean_pgd,
        S_adv_pgd,
        cos_shap_pgd,
        euc_shap_pgd,
        kl_shap_pgd
    )

    plot_roc(
        y_shap_pgd,
        kl_shap_pgd_scores,
        "SHAP + PGD KL Drift ROC",
        "roc_shap_pgd_kl.png"
    )

# ===============================
# SHAP + FGSM KL ROC
# ===============================
if RUN_FGSM:
    y_shap_fgsm, cos_shap_fgsm_scores, euc_shap_fgsm_scores, kl_shap_fgsm_scores = prepare_labels_and_scores_with_kl(
        S_clean_fgsm,
        S_adv_fgsm,
        cos_shap_fgsm,
        euc_shap_fgsm,
        kl_shap_fgsm
    )

    plot_roc(
        y_shap_fgsm,
        kl_shap_fgsm_scores,
        "SHAP + FGSM KL Drift ROC",
        "roc_shap_fgsm_kl.png"
    )


