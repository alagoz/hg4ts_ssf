"""Stochastic Hierarchy Induction Module"""
import numpy as np
from sklearn.model_selection import train_test_split

from hg_ssf import hdcSSF
from utils import (_handle_small_dataset_fallback,
                   _split_data_with_fallback)

class StochasticHierarchyInductor:
    """
    Stochastic Hierarchy Induction for generating multiple hierarchical structures
    and selecting the best one based on validation performance.
    """
    
    def __init__(self, 
                 base_estimator_hg=None,
                 base_estimator_eval=None,
                 splitting_function='potr',
                 n_iter=3,
                 n_iter_val=1,
                 n_val=4,
                 val_split_mode = 'original',
                 tree_selection_metric='acc',
                 avoid_duplicates=True,
                 multi_process=True,
                 random_state=None,
                 verbose=0):
        """
        Parameters
        ----------
        base_estimator_hg : estimator
            Base estimator for hierarchy generation
        base_estimator_hg : estimator
            Base estimator for tree evaluation
        splitting_function : str
            Splitting function for hierarchy generation
        n_iter : int
            Number of stochastic hierarchy iterations
        n_iter_val : int
            Number of ncv iterations. Adaptive if None
        n_val : int
            Number of folds for hg and nested cross-validation
        tree_selection_metric : str
            Metric for tree selection ('acc', 'f1', 'balacc')
        avoid_duplicates : bool
            Whether to avoid duplicate tree structures
        random_state : int
            Random state for reproducibility
        verbose : int
            Verbosity level
        """
        self.base_estimator_hg = base_estimator_hg
        self.base_estimator_eval = base_estimator_eval
        self.splitting_function = splitting_function
        self.n_iter = n_iter
        self.n_iter_val = n_iter_val
        self.n_val = n_val
        self.val_split_mode = val_split_mode
        self.tree_selection_metric = tree_selection_metric
        self.avoid_duplicates = avoid_duplicates
        self.multi_process=multi_process
        self.random_state = random_state
        self.verbose = verbose
        
        
    def induce_hierarchies(self, X, y, base_estimator_he, validation_data=None):
        """
        Perform stochastic hierarchy induction
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training labels
        base_estimator_he : estimator
            Base estimator for hierarchy exploitation
        validation_data : tuple, optional
            (X_val, y_val) for validation
            
        Returns
        -------
        best_hierarchy : dict
            Dictionary containing the best hierarchy structure
        all_hierarchies : list
            List of all generated hierarchies with their scores
        """
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.verbose >= 1:
            print(f"Stochastic Hierarchy Induction with {self.n_iter} iterations")
        
        best_score = -1
        best_hierarchy = None
        all_hierarchies = []
        tree_hashes = set()
        
        for i in range(self.n_iter):
            if self.verbose >= 2:
                print(f"  Iteration {i+1}/{self.n_iter}")
            
            try:
                X_train, X_val, y_train, y_val = self._create_validation_split(X, y)
                
                # Generate hierarchy
                hierarchy = self._generate_hierarchy(X_train, y_train, X_val, y_val)
                if hierarchy is None:
                    continue
                
                # Check for duplicates
                if self.avoid_duplicates:
                    tree_hash = self._get_tree_hash(hierarchy)
                    if tree_hash in tree_hashes:
                        if self.verbose >= 3:
                            print("    Skipping duplicate tree")
                        continue
                    tree_hashes.add(tree_hash)
                
                # Evaluate hierarchy
                score, eval_scores = self._evaluate_hierarchy(
                    hierarchy, X, y, base_estimator_he, X_val, y_val
                )
                
                hierarchy_info = {
                    'parent_nodes': hierarchy['parent_nodes'],
                    'linkage_matrix': hierarchy['linkage_matrix'],
                    'score': score,
                    'eval_scores': eval_scores,
                    'iteration': i
                }
                
                all_hierarchies.append(hierarchy_info)
                
                # Update best hierarchy
                if score > best_score:
                    best_score = score
                    best_hierarchy = hierarchy_info
                    if self.verbose >= 2:
                        print(f"    New best score: {best_score:.4f}")
                        
            except Exception as e:
                if self.verbose >= 1:
                    print(f"    Iteration {i+1} failed: {e}")
                continue
        
        if best_hierarchy is None:
            raise RuntimeError("No valid hierarchy generated during induction")
        
        if self.verbose >= 1:
            print(f"Best hierarchy selected with score: {best_score:.4f}")
        
        return best_hierarchy, all_hierarchies
    
    def _create_validation_split(self, X, y):
        # Create validation split
        if self.val_split_mode == 'fallback':
            dset_val = _split_data_with_fallback(X, y, shuffle_=True, verbose=self.verbose)
            X_train, X_val, y_train, y_val = dset_val
            
        elif self.val_split_mode == 'adaptive':
            adaptive_n_val = _handle_small_dataset_fallback(X, y, self.n_val)            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=1/adaptive_n_val, stratify=y, random_state=self.random_state
            )
            self.adaptive_n_val = adaptive_n_val
            
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=1/self.n_val, stratify=y, random_state=self.random_state
            )
            
        return X_train, X_val, y_train, y_val
            
    def _generate_hierarchy(self, X_train, y_train, X_val, y_val):
        """Generate a single hierarchy using the splitting function"""
        try:
            model = hdcSSF(
                dset=(X_train, X_val, y_train, y_val),
                ssf=self.splitting_function,
                base_estimator=self.base_estimator_hg,
                eval_metric='acc'
            )
            Z, parent_nodes = model.fit()
            
            if self.verbose >= 3:
                from utils import plot_dendrogram
                plot_dendrogram(Z,close_all=1)
                
            return {
                'parent_nodes': parent_nodes,
                'linkage_matrix': Z,
                'model': model
            }
        
        except Exception as e:
            if self.verbose >= 3:
                print(f"    Hierarchy generation failed: {e}")
            return None
    
    def _evaluate_hierarchy(self, hierarchy, X, y, base_estimator_he, X_val, y_val):
        """Evaluate a hierarchy using nested cross-validation"""
        from he_binary_tree import BinaryTreeClassifier
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
        
        scores = []
        
        for ki in range(self.n_iter_val):
            try:
                # Create validation split for this fold
                X_train_ki, X_val_ki, y_train_ki, y_val_ki = self._create_validation_split(X, y)
                
                # Build and evaluate tree
                tree = BinaryTreeClassifier(
                    pnodes=hierarchy['parent_nodes'],
                    y_train=y_train_ki,
                    y_test=y_val_ki,
                    link_mat=hierarchy['linkage_matrix']
                )
                
                tree.fit(base_estimator_he, X_train_ki, he_type='lcpn',
                        rseed=self.random_state, multi_process=self.multi_process)
                
                # Evaluate
                y_pred = tree.predict(X_val_ki, he_type='lcpn+')
                
                # Calculate score based on selection metric
                if self.tree_selection_metric == 'acc':
                    score = accuracy_score(y_val_ki, y_pred)
                elif self.tree_selection_metric == 'f1':
                    score = f1_score(y_val_ki, y_pred, average='macro')
                elif self.tree_selection_metric == 'balacc':
                    score = balanced_accuracy_score(y_val_ki, y_pred)
                else:
                    score = accuracy_score(y_val_ki, y_pred)
                
                scores.append(score)
                
            except Exception as e:
                if self.verbose >= 3:
                    print(f"      Fold {ki+1} failed: {e}")
                scores.append(0.0)
        
        mean_score = np.mean(scores) if scores else 0.0
        return mean_score, scores
    
    def _get_tree_hash(self, hierarchy):
        """Generate hash for tree structure to detect duplicates"""
        import hashlib
        tree_repr = str(hierarchy['parent_nodes']) + str(hierarchy['linkage_matrix'].tolist())
        return hashlib.md5(tree_repr.encode()).hexdigest()