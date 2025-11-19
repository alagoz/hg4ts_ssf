"""
Stochastic Hierarchy Induction (SHI) Demo
=========================================

This demo showcases the proposed Stochastic Hierarchy Induction (SHI) framework 
for automated hierarchy generation (HG) and exploitation (HE) in Time Series 
Classification (TSC).

The SHI framework introduces Stochastic Splitting Functions (SSFs)—'potr', 
'srtr', and 'lsoo'—to construct discriminative, classifier-informed hierarchical 
structures directly from flat label sets. Hierarchies are exploited using the 
extended Local Classifier Per Node strategy (LCPN+), enabling performance-driven 
hierarchical modeling.

The script performs an end-to-end evaluation of SHI-based Hierarchical 
Classification (HC) against traditional Flat Classification (FC), using 
MiniRocket-transformed representations and efficient linear classifiers.

Users can configure:
  • The stochastic splitting function (e.g., 'srtr', 'potr', 'lsoo')
  • Base estimators for hierarchy generation (est_hg) and exploitation (est_he)
  • Dataset name, random seed, and number of stochastic iterations
  • Cross-validation and evaluation options

This demonstration highlights the generalization behavior, efficiency, and 
predictive advantages of SHI-based hierarchical modeling compared to 
flat baselines on benchmark time series datasets.

Author: Celal Alagoz
License: MIT
"""

import numpy as np
from time import time
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV as Ridge
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, 
                             roc_auc_score, log_loss)
from aeon.datasets import load_classification
from aeon.transformations.collection.convolution_based import MiniRocket

# Import custom modules
from utils import sort_class_ids, to_one_hot
from shi import StochasticHierarchyInductor
from he_binary_tree import BinaryTreeClassifier


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
def load_and_prepare_data(dataset_name, test_size=0.2, random_state=42):
    """Load and prepare dataset for classification."""
    print(f"📊 Loading dataset: {dataset_name}")
    X, y, meta = load_classification(dataset_name, return_metadata=True)
    X = X.squeeze()
    y = sort_class_ids(y).astype(int)
    
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def transform_data(X_train, X_test, scaler=True):
    """Apply MiniRocket transformation and optional scaling."""
    print("🔄 Applying MiniRocket transformation...")
    model_trans = MiniRocket(n_jobs=-1, random_state=42)
    
    X_train_trans = model_trans.fit_transform(X_train)
    X_test_trans = model_trans.transform(X_test)
    
    if scaler:
        scaler = StandardScaler(with_mean=False)
        X_train_trans = scaler.fit_transform(X_train_trans)
        X_test_trans = scaler.transform(X_test_trans)
    
    print(f"   Transformed - Train: {X_train_trans.shape}, Test: {X_test_trans.shape}")
    return X_train_trans, X_test_trans


# =============================================================================
# Evaluation Utilities
# =============================================================================
def evaluate_predictions(y_true, y_pred, y_proba, method_name):
    """Compute evaluation metrics for classification results."""
    results = {
        'method': method_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }
    
    # Probability-based metrics
    if y_proba is not None:
        try:
            results['log_loss'] = log_loss(y_true, y_proba)
        except Exception:
            results['log_loss'] = np.nan
            
        try:
            if len(np.unique(y_true)) > 2:
                results['auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            else:
                results['auc_ovr'] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            results['auc_ovr'] = np.nan
    else:
        results['log_loss'] = np.nan
        results['auc_ovr'] = np.nan
    
    return results


# =============================================================================
# Hierarchical and Flat Classification
# =============================================================================
def run_hierarchical_classification(
    X_train, X_test, y_train, y_test, 
    est_hg=None, est_he=None, 
    splitting_function='srtr', n_iter=3, 
    random_state=42
):
    """Run hierarchical classification using the LCPN+ strategy."""
    print("\n🌳 Running Hierarchical Classification (HC)...")
    
    # Default estimators
    if est_hg is None:
        est_hg = Ridge(alphas=np.logspace(-3, 3, 10))
    if est_he is None:
        est_he = Ridge(alphas=np.logspace(-3, 3, 10))
    
    durations = {}
    
    # Step 1: Hierarchy Generation (HG)
    print(f"   Step 1: Hierarchy Induction using '{splitting_function}' ...")
    inductor = StochasticHierarchyInductor(
        base_estimator_hg=clone(est_hg),
        splitting_function=splitting_function,
        n_iter=n_iter,
        val_split_mode='adaptive',
        multi_process=False,
        random_state=random_state,
        verbose=1
    )
    
    t0 = time()
    best_hierarchy, all_hierarchies = inductor.induce_hierarchies(
        X_train, y_train, clone(est_he)
    )
    durations['hierarchy_induction'] = time() - t0
    print(f"      Generated {len(all_hierarchies)} hierarchies")
    
    # Step 2: Hierarchy Exploitation (HE)
    print("   Step 2: Training Hierarchical Exploitation Model (LCPN+)...")
    tree = BinaryTreeClassifier(
        pnodes=best_hierarchy['parent_nodes'],
        y_train=y_train,
        y_test=y_test,
        link_mat=best_hierarchy['linkage_matrix']
    )
    
    t0 = time()
    tree.fit(clone(est_he), X_train, he_type='lcpn', rseed=random_state, multi_process=True)
    durations['classifier_training'] = time() - t0
    
    # Step 3: Prediction
    print("   Step 3: Predicting with LCPN+...")
    t0 = time()
    y_pred = tree.predict(X_test, he_type='lcpn+')
    y_proba = tree.predict_proba(X_test, he_type='lcpn+')
    durations['prediction'] = time() - t0
    
    results = evaluate_predictions(y_test, y_pred, y_proba, f"HC-{splitting_function}")
    results['durations'] = durations
    results['total_duration'] = sum(durations.values())
    
    print(f"      Completed in {results['total_duration']:.2f}s")
    return results


def run_flat_classification(X_train, X_test, y_train, y_test, est_fc=None):
    """Run flat classification for baseline comparison."""
    print("\n⚡ Running Flat Classification (FC)...")
    
    if est_fc is None:
        est_fc = Ridge(alphas=np.logspace(-3, 3, 10))
    
    durations = {}
    
    t0 = time()
    fc_model = clone(est_fc)
    fc_model.fit(X_train, y_train)
    durations['training'] = time() - t0
    
    t0 = time()
    y_pred = fc_model.predict(X_test)
    try:
        y_proba = fc_model.predict_proba(X_test)
    except Exception:
        y_proba = to_one_hot(y_pred)
    durations['prediction'] = time() - t0
    
    results = evaluate_predictions(y_test, y_pred, y_proba, "FC")
    results['durations'] = durations
    results['total_duration'] = sum(durations.values())
    
    print(f"   Completed in {results['total_duration']:.2f}s")
    return results


# =============================================================================
# Comparison Output
# =============================================================================
def print_comparison_table(hc_results, fc_results):
    """Display performance and timing comparison between HC and FC."""
    print("\n" + "="*80)
    print("📊 COMPARISON: Hierarchical vs Flat Classification")
    print("="*80)
    
    metrics = ['accuracy', 'f1_macro', 'balanced_accuracy', 'auc_ovr', 'log_loss']
    print(f"{'Metric':<20} {'HC':<10} {'FC':<10} {'Δ (HC−FC)':<12} {'Best':<8}")
    print("-" * 80)
    
    for metric in metrics:
        hc_val = hc_results.get(metric, np.nan)
        fc_val = fc_results.get(metric, np.nan)
        
        if metric == 'log_loss':
            diff = fc_val - hc_val
            best = "HC" if diff > 0 else "FC" if diff < 0 else "Tie"
        else:
            diff = hc_val - fc_val
            best = "HC" if diff > 0 else "FC" if diff < 0 else "Tie"
        
        print(f"{metric:<20} {hc_val:<10.4f} {fc_val:<10.4f} {diff:+.4f}      {best:<8}")
    
    print("\n" + "="*80)
    print("⏱️  TIME COMPARISON")
    print("="*80)
    hc_total = hc_results['total_duration']
    fc_total = fc_results['total_duration']
    print(f"HC Total: {hc_total:.2f}s, FC Total: {fc_total:.2f}s (HC/FC = {hc_total/fc_total:.2f}×)")


# =============================================================================
# Main Execution
# =============================================================================
def main():
    print("="*80)
    print("🌳 HIERARCHICAL vs FLAT CLASSIFICATION DEMO")
    print("="*80)
    
    # --- CONFIGURATION ---
    DATASET = "Tools"            # Dataset name from UCR archive
    RANDOM_STATE = 42
    SPLITTING_FUNCTION = "srtr"  # Options: "srtr", "potr", "lsoo"
    N_ITER = 3
    EST_HG = Ridge(alphas=np.logspace(-3, 3, 10))
    EST_HE = Ridge(alphas=np.logspace(-3, 3, 10))
    EST_FC = Ridge(alphas=np.logspace(-3, 3, 10))
    
    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data(DATASET, random_state=RANDOM_STATE)
        X_train_trans, X_test_trans = transform_data(X_train, X_test)
        
        hc_results = run_hierarchical_classification(
            X_train_trans, X_test_trans, y_train, y_test,
            est_hg=EST_HG, est_he=EST_HE,
            splitting_function=SPLITTING_FUNCTION, n_iter=N_ITER,
            random_state=RANDOM_STATE
        )
        
        fc_results = run_flat_classification(X_train_trans, X_test_trans, y_train, y_test, EST_FC)
        print_comparison_table(hc_results, fc_results)
        
        print("\n🎉 Demo completed successfully!")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check dependencies or dataset availability.")


if __name__ == "__main__":
    main()
