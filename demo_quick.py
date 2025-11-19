"""Quick Usage Example: Stochastic Hierarchy Induction (SHI)
=============================================================

This quick demo compares Hierarchical Classification (HC) using the
proposed Stochastic Hierarchy Induction (SHI) framework against
Flat Classification (FC) on a UCR time series dataset.

Supports multiple transform models: miniR, quant, or cfire.

Author: Celal Alagoz
License: MIT
"""

import numpy as np
from time import time
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from aeon.datasets import load_classification

# Transformation and classification models
from aeon.transformations.collection.convolution_based import MiniRocket
from aeon.transformations.collection.interval_based import QUANTTransformer
from sklearn.linear_model import RidgeClassifierCV as Ridge
from sklearn.ensemble import ExtraTreesClassifier

# Custom modules
from utils import sort_class_ids
from shi import StochasticHierarchyInductor
from he_binary_tree import BinaryTreeClassifier
from crossfire import CFIRE

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_NAME = "Tools"           # Any dataset from UCR archive
TRANSFORM_MODEL = "cfire"      # Options: "MiniRocket", "Quant", "Cfire", "None"
SPLITTING_FUNCTION = "srtr"         # Options: "potr", "srtr", "lsoo"
N_ITER = 3                          # Number of stochastic iterations
SCALED_TRANSFORM = True             # Whether to apply feature scaling
RANDOM_STATE = 42                   # Random seed for reproducibility
TEST_SIZE = 0.2                     # Train-test split ratio
VERBOSE = 1                         # 0: silent, 1: info, 2: debug

# Classifier configurations
CLF_RIDGE = Ridge(alphas=np.logspace(-3, 3, 10))
CLF_ETC = ExtraTreesClassifier(n_estimators=200, n_jobs=1, random_state=RANDOM_STATE)

EST_HG = CLF_ETC                  # Estimator for hierarchy generation
EST_HE = CLF_ETC                  # Estimator for hierarchy exploitation
EST_FC = CLF_ETC                  # Estimator for flat baseline

# ============================================================
# DATA LOADING & TRANSFORMATION
# ============================================================
print(f"\n📂 Loading dataset: {DATASET_NAME}")
X, y, meta = load_classification(DATASET_NAME, return_metadata=True)
X = X.squeeze()
y = sort_class_ids(y).astype(int)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Select transformation model
if TRANSFORM_MODEL == "miniR":
    print("\n🔄 Applying MiniRocket transformation...")
    model_trans = MiniRocket(n_jobs=-1, random_state=RANDOM_STATE)
elif TRANSFORM_MODEL == "quant":
    print("\n📈 Applying Quant transformation...")
    model_trans = QUANTTransformer(n_jobs=-1, random_state=RANDOM_STATE)
elif TRANSFORM_MODEL == "cfire":
    print("\n🔥 Applying Crossfire transformation...")
    model_trans = CFIRE(norms = 1,
                   stats = 1,
                   series= 1,
                   temp = 1,
                   multiprocessing=True,
                   )
elif TRANSFORM_MODEL == "None":
    print("\n⚙️  Using raw features (no transformation)...")
    model_trans = None
else:
    raise ValueError(f"Unknown transform model: {TRANSFORM_MODEL}")

# Apply transformation
if model_trans is not None:
    X_train = model_trans.fit_transform(X_train)
    X_test = model_trans.fit_transform(X_test)

# Optional scaling
if SCALED_TRANSFORM:
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# ============================================================
# STEP 1: STOCHASTIC HIERARCHY INDUCTION
# ============================================================
print(f"\n🌳 Inducing Stochastic Hierarchy (SSF = '{SPLITTING_FUNCTION}')...")
inductor = StochasticHierarchyInductor(
    base_estimator_hg=clone(EST_HG),
    splitting_function=SPLITTING_FUNCTION,
    n_iter=N_ITER,
    val_split_mode="adaptive",
    multi_process=False,
    random_state=RANDOM_STATE,
    verbose=VERBOSE,
)

t0 = time()
best_hierarchy, _ = inductor.induce_hierarchies(X_train, y_train, clone(EST_HE))
DUR_HG_FIT = time() - t0
print(f"   Hierarchy induction completed in {DUR_HG_FIT:.2f}s")

# Optional: visualize hierarchy
try:
    from utils import plot_dendrogram
    plot_dendrogram(best_hierarchy["linkage_matrix"], close_all=1)
except Exception:
    pass


# ============================================================
# STEP 2: TRAIN HIERARCHICAL CLASSIFIER (LCPN+)
# ============================================================
print("\n⚙️  Training hierarchical classifier (LCPN+)...")
tree = BinaryTreeClassifier(
    pnodes=best_hierarchy["parent_nodes"],
    y_train=y_train,
    y_test=y_test,
    link_mat=best_hierarchy["linkage_matrix"],
)

t0 = time()
tree.fit(EST_HE, X_train, he_type="lcpn", rseed=RANDOM_STATE, multi_process=True)
DUR_HE_FIT = time() - t0
print(f"   HC training completed in {DUR_HE_FIT:.2f}s")

t0 = time()
y_pred_hc = tree.predict(X_test, he_type="lcpn+")
DUR_HE_PRED = time() - t0


# ============================================================
# STEP 3: FLAT CLASSIFICATION BASELINE
# ============================================================
print("\n⚡ Running flat classification (FC baseline)...")
t0 = time()
EST_FC.fit(X_train, y_train)
DUR_FC_FIT = time() - t0

t0 = time()
y_pred_fc = EST_FC.predict(X_test)
DUR_FC_PRED = time() - t0


# ============================================================
# STEP 4: PERFORMANCE SUMMARY
# ============================================================
def summarize(name, y_true, y_pred, dur_train, dur_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_true, y_pred)
    total = dur_train + dur_pred
    print(f"\n🧩 {name} Results:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Macro: {f1:.4f}")
    print(f"   Balanced Accuracy: {bacc:.4f}")
    print(f"   Train Time: {dur_train:.2f}s | Test Time: {dur_pred:.2f}s | Total: {total:.2f}s")
    return acc, total


acc_hc, time_hc = summarize("Hierarchical (HC-lcpn+)", y_test, y_pred_hc, DUR_HE_FIT, DUR_HE_PRED)
acc_fc, time_fc = summarize("Flat (FC)", y_test, y_pred_fc, DUR_FC_FIT, DUR_FC_PRED)

print("\n" + "="*60)
print("📊 COMPARISON SUMMARY")
print("="*60)
print(f"Δ Accuracy (HC - FC): {acc_hc - acc_fc:+.4f}")
print(f"Time Ratio (HC/FC): {time_hc / time_fc:.2f}x")
print("="*60)
