"""
Hierarchical vs Flat Classification Demo
========================================

A comprehensive comparison between Hierarchical Classification (HC) 
and Flat Classification (FC) on time series datasets.

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

# Import your custom modules
from utils import sort_class_ids, to_one_hot
from shi import StochasticHierarchyInductor
from he_binary_tree import BinaryTreeClassifier


def load_and_prepare_data(dataset_name, test_size=0.2, random_state=42):
    """Load and prepare dataset for classification."""
    print(f"📊 Loading dataset: {dataset_name}")
    X, y, meta = load_classification(dataset_name, return_metadata=True)
    X = X.squeeze()
    y = sort_class_ids(y)
    y = y.astype(int)
    
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data
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


def evaluate_predictions(y_true, y_pred, y_proba, method_name):
    """Comprehensive evaluation of predictions."""
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
        except:
            results['log_loss'] = np.nan
            
        try:
            if len(np.unique(y_true)) > 2:
                results['auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            else:
                results['auc_ovr'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            results['auc_ovr'] = np.nan
    else:
        results['log_loss'] = np.nan
        results['auc_ovr'] = np.nan
    
    return results


def run_hierarchical_classification(X_train, X_test, y_train, y_test, random_state=42):
    """Run hierarchical classification with lcpn+ strategy."""
    print("\n🌳 Running Hierarchical Classification (HC)...")
    
    # Configuration
    est_hg = Ridge(alphas=np.logspace(-3, 3, 10))
    est_he = Ridge(alphas=np.logspace(-3, 3, 10))
    
    durations = {}
    results = {}
    
    # Step 1: Hierarchy Induction
    print("   Step 1: Stochastic Hierarchy Induction...")
    inductor = StochasticHierarchyInductor(
        base_estimator_hg=clone(est_hg),
        splitting_function='srtr',
        n_iter=3,
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
    
    # Plot dendrogram
    try:
        from utils import plot_dendrogram
        plot_dendrogram(best_hierarchy['linkage_matrix'], close_all=1)
        print("📊 Dendrogram plotted")
    except:
        print("⚠️  Could not plot dendrogram")
    
    # Step 2: Train HC Classifier
    print("   Step 2: Training Hierarchical Classifier...")
    tree = BinaryTreeClassifier(
        pnodes=best_hierarchy['parent_nodes'],
        y_train=y_train,
        y_test=y_test,
        link_mat=best_hierarchy['linkage_matrix']
    )
    
    t0 = time()
    tree.fit(clone(est_he), X_train, he_type='lcpn', rseed=random_state, multi_process=True)
    durations['classifier_training'] = time() - t0
    
    # Step 3: HC Prediction (lcpn+)
    print("   Step 3: Prediction with lcpn+...")
    t0 = time()
    y_pred_hc = tree.predict(X_test, he_type='lcpn+')
    y_pred_proba_hc = tree.predict_proba(X_test, he_type='lcpn+')
    durations['prediction'] = time() - t0
    
    # Evaluate
    results = evaluate_predictions(y_test, y_pred_hc, y_pred_proba_hc, "HC-lcpn+")
    results['durations'] = durations
    results['total_duration'] = sum(durations.values())
    
    print(f"      Completed in {results['total_duration']:.2f}s")
    return results


def run_flat_classification(X_train, X_test, y_train, y_test):
    """Run flat classification for comparison."""
    print("\n⚡ Running Flat Classification (FC)...")
    
    est_fc = Ridge(alphas=np.logspace(-3, 3, 10))
    durations = {}
    
    # Training
    t0 = time()
    fc_model = clone(est_fc)
    fc_model.fit(X_train, y_train)
    durations['training'] = time() - t0
    
    # Prediction
    t0 = time()
    y_pred_fc = fc_model.predict(X_test)
    try:
        y_pred_proba_fc = fc_model.predict_proba(X_test)
    except:
        y_pred_proba_fc = to_one_hot(y_pred_fc)
    durations['prediction'] = time() - t0
    
    # Evaluate
    results = evaluate_predictions(y_test, y_pred_fc, y_pred_proba_fc, "FC")
    results['durations'] = durations
    results['total_duration'] = sum(durations.values())
    
    print(f"   Completed in {results['total_duration']:.2f}s")
    return results


def print_comparison_table(hc_results, fc_results):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE COMPARISON: HC vs FC")
    print("="*80)
    
    # Metrics comparison
    metrics = ['accuracy', 'f1_macro', 'balanced_accuracy', 'auc_ovr', 'log_loss']
    metric_names = ['Accuracy', 'F1-Macro', 'Balanced Acc', 'AUC-OVR', 'Log Loss']
    
    print(f"{'Metric':<15} {'HC-lcpn+':<10} {'FC':<10} {'Difference':<12} {'Best':<8}")
    print("-" * 80)
    
    for metric, name in zip(metrics, metric_names):
        hc_val = hc_results.get(metric, np.nan)
        fc_val = fc_results.get(metric, np.nan)
        
        if metric == 'log_loss':
            # Lower is better for log loss
            diff = fc_val - hc_val  # Positive means HC is better
            best = "HC" if diff > 0 else "FC" if diff < 0 else "Tie"
        else:
            # Higher is better for other metrics
            diff = hc_val - fc_val  # Positive means HC is better
            best = "HC" if diff > 0 else "FC" if diff < 0 else "Tie"
        
        print(f"{name:<15} {hc_val:<10.4f} {fc_val:<10.4f} {diff:+.4f}      {best:<8}")
    
    # Timing comparison
    print("\n" + "="*80)
    print("⏱️  COMPUTATION TIME COMPARISON")
    print("="*80)
    
    hc_total = hc_results['total_duration']
    fc_total = fc_results['total_duration']
    time_ratio = hc_total / fc_total
    
    print(f"{'Method':<15} {'Total Time':<12} {'Training':<12} {'Prediction':<12}")
    print("-" * 80)
    print(f"{'HC-lcpn+':<15} {hc_total:<12.2f} {hc_results['durations']['classifier_training']:<12.2f} {hc_results['durations']['prediction']:<12.2f}")
    print(f"{'FC':<15} {fc_total:<12.2f} {fc_results['durations']['training']:<12.2f} {fc_results['durations']['prediction']:<12.2f}")
    print(f"\nTime ratio (HC/FC): {time_ratio:.2f}x")
    
    # Performance summary
    hc_acc = hc_results['accuracy']
    fc_acc = fc_results['accuracy']
    acc_improvement = hc_acc - fc_acc
    
    print("\n" + "="*80)
    print("🎯 PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Accuracy: HC-lcpn+ = {hc_acc:.4f}, FC = {fc_acc:.4f}")
    print(f"Improvement: {acc_improvement:+.4f} ({acc_improvement/hc_acc*100:+.1f}%)")
    
    if acc_improvement > 0.01:
        print("✅ RECOMMENDATION: Use Hierarchical Classification")
    elif acc_improvement > 0:
        print("⚖️  RECOMMENDATION: HC shows modest improvement, consider based on requirements")
    else:
        print("⚠️  RECOMMENDATION: Flat Classification is sufficient")


def main():
    """Main demo function."""
    print("="*80)
    print("🌳 HIERARCHICAL vs FLAT CLASSIFICATION DEMO")
    print("="*80)
    
    # Configuration
    DATASET = "Tools"  # Change this to test other datasets
    RANDOM_STATE = 42
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            DATASET, random_state=RANDOM_STATE
        )
        
        # Transform data
        X_train_trans, X_test_trans = transform_data(X_train, X_test, scaler=True)
        
        # Run hierarchical classification
        hc_results = run_hierarchical_classification(
            X_train_trans, X_test_trans, y_train, y_test, RANDOM_STATE
        )
        
        # Run flat classification
        fc_results = run_flat_classification(
            X_train_trans, X_test_trans, y_train, y_test
        )
        
        # Print comprehensive comparison
        print_comparison_table(hc_results, fc_results)
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check your installation and dataset availability.")


if __name__ == "__main__":
    main()