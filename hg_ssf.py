import numpy as np
import warnings
import copy
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from aeon.classification.convolution_based import MiniRocketClassifier
from utils import encode_super_labels, to_one_hot, monotonize_rescale_
from sklearn.base import clone
import hashlib

from he_binary_tree import HierNode

class hdcSSF():
    def __init__(self, dset=None, base_estimator=None, eval_metric='acc', 
                 ssf='potr', random_state=None, n_jobs=-1):
        """
        Hierarchical Divisive Clustering with Stochastic Splitting Functions
        
        Parameters
        ----------
        dset : tuple, optional
            Dataset tuple (X_train, y_train, X_test, y_test) or (X, y)
        base_estimator : object, optional
            A scikit-learn compatible classifier. If None, uses Minirocket + RidgeClassifier.
        eval_metric : {'acc', 'f1', 'balacc', 'auc'}
            Evaluation metric for split quality
        ssf : {'potr', 'srtr', 'lsoo'}
            The Stochastic Splitting Function to use.        
        random_state : int
            Seed for the random number generator.
        n_jobs : int
            Number of jobs to run in parallel.
        """
        self.dset = dset
        self.base_estimator = base_estimator
        self.eval_metric = eval_metric
        self.ssf = ssf
        self.random_state = random_state
        self.n_jobs = n_jobs        
        self.classes_ = None
        self.Z = None
        self.nodes_ = None
        self.is_fitted = False
        self.tree_hash = None
        self.verbose=False
        
        if dset is not None:
            if len(dset) == 4:
                self.X_train, self.X_test, self.y_train, self.y_test = dset
                self.X = np.concatenate([self.X_train, self.X_test])
                self.y = np.concatenate([self.y_train, self.y_test])
            elif len(dset) == 2:
                self.X, self.y = dset
            else:
                raise ValueError("dset must be (X_train, y_train, X_test, y_test) or (X, y)")
            
            self.classes_ = np.unique(self.y)
            self.n_object = len(self.classes_)
            self.n_clusters = self.n_object - 1
            
        np.random.seed(random_state)
        warnings.filterwarnings("ignore")
                            
    def _get_estimator(self):
        """Get a clone of the base estimator. Uses Minirocket if none is provided."""
        if self.base_estimator is None:
            return MiniRocketClassifier(random_state=self.random_state)
        else:
            return clone(self.base_estimator)
        
    def _split_data(self, X, y, groups):
        """Get data for a specific set of classes/groups."""
        idx = np.isin(y, groups)
        return X[idx], y[idx]
    
    def _update_score_and_groups(self, score, max_score, cand_0, cand_1, best_0, best_1):
        """Routine to update score and groups."""
        if score > max_score:
            return score, cand_0.copy(), cand_1.copy()
        else:
            return max_score, best_0, best_1
        
    def _evaluate_split_(self, X, y, groups_0, groups_1):
        """Evaluate a split using eval metric score."""
        X_0, y_0 = self._split_data(X, y, groups_0)
        X_1, y_1 = self._split_data(X, y, groups_1)
        
        if len(X_0) == 0 or len(X_1) == 0:
            return 0.0 # Avoid splits that create empty sets
            
        X_comb = np.concatenate((X_0, X_1))
        y_comb = np.concatenate((y_0, y_1))
        
        # Create binary labels for the two meta-groups
        y_binary = np.concatenate((np.zeros(len(y_0)), np.ones(len(y_1))))
        
        temp_estimator = self._get_estimator()
        
        try:
            temp_estimator.fit(X_comb, y_binary)
            y_pred = temp_estimator.predict(X_comb)
            y_pred_proba = temp_estimator.predict_proba(X_comb) if hasattr(temp_estimator, 'predict_proba') else None
            
            if self.eval_metric == 'acc':
                score = accuracy_score(y_binary, y_pred)
            elif self.eval_metric == 'f1':
                score = f1_score(y_binary, y_pred, average='binary')
            elif self.eval_metric == 'balacc':
                score = balanced_accuracy_score(y_binary, y_pred)
            elif self.eval_metric == 'auc':
                if y_pred_proba is not None and len(np.unique(y_binary)) == 2:
                    score = roc_auc_score(y_binary, y_pred_proba[:, 1])
                else:
                    score = 0.5  # Neutral score for undefined AUC
            else:
                raise ValueError(f"Unknown metric: {self.eval_metric}")
                
            return score
            
        except Exception as e:
            print(f"Split evaluation failed: {e}")
            return 0.0
        
    def _evaluate_split(self, super_classes=[]):
        """Evaluate a split using eval metric score."""
        
        # Dataset with flat labels
        x_tr, x_te, y_tr, y_te = self.dset
        
        # Dataset with super labels
        y_tr, x_tr = encode_super_labels(super_classes, y_tr, x_tr)
        y_te, x_te = encode_super_labels(super_classes, y_te, x_te)
               
        temp_estimator = self._get_estimator()
        
        try:
            temp_estimator.fit(x_tr, y_tr)
            y_pred = temp_estimator.predict(x_te)
            if hasattr(temp_estimator, 'predict_proba'):
                y_pred_proba = temp_estimator.predict_proba(x_te) 
            else: 
                y_pred_proba = to_one_hot(y_pred)
            
            if self.eval_metric == 'acc':
                score = accuracy_score(y_te, y_pred)
            elif self.eval_metric == 'f1':
                score = f1_score(y_te, y_pred, average='macro')
            elif self.eval_metric == 'balacc':
                score = balanced_accuracy_score(y_te, y_pred)
            elif self.eval_metric == 'auc':
                if y_pred_proba is not None and len(np.unique(y_te)) == 2:
                    score = roc_auc_score(y_te, y_pred_proba[:, 1])
                else:
                    score = 0.5  # Neutral score for undefined AUC
            else:
                raise ValueError(f"Unknown metric: {self.eval_metric}")
                
            return score
            
        except Exception as e:
            print(f"Split evaluation failed: {e}")
            return 0.0
        
    def _potr(self, classes):
        """Pick-One-Then-Regroup."""
        np.random.seed(self.random_state)
        
        # Pick one class randomly
        j = np.random.randint(0, len(classes))
        C_0 = [classes[j]]
        C_1 = [c for i, c in enumerate(classes) if i != j]
        
        # Initial evaluation
        max_score = self._evaluate_split(super_classes=[C_0, C_1])
        best_C0, best_C1 = C_0, C_1
    
        # Iterate through remaining classes in C_1
        iter_classes = C_1.copy()
        for c in iter_classes:
            
            if max_score >= 0.99: # Near-perfect score, break early
                if self.verbose: 
                    print('Maxiumum performance reached')
                break
            
            # Create new split by moving c from C_1 to C_0
            C_0_prime = C_0 + [c]
            C_1_prime = [elem for elem in C_1 if elem != c]
            
            # Evaluate the new split
            score_temp = self._evaluate_split(super_classes=[C_0_prime, C_1_prime])
            
            # Update if score improves and C_1 has more than 1 element
            if score_temp > max_score and len(C_1_prime) > 1:
                max_score = score_temp
                best_C0, best_C1 = C_0_prime, C_1_prime
                                
                if self.verbose: 
                    print(f'{c} passed from cluster 1 to cluster 0')
        
        return best_C0, best_C1, max_score

    def _srtr(self, classes):
        """Split-Randomly-Then-Regroup."""
        np.random.seed(self.random_state)
        shuffled = classes.copy()
        np.random.shuffle(shuffled)
        # j = np.random.randint(1, len(shuffled))
        j = len(shuffled)//2
        C_0 = list(shuffled[:j])
        C_1 = list(shuffled[j:])
        
        max_score = self._evaluate_split(super_classes=[C_0, C_1])
        best_C0, best_C1 = C_0, C_1
        
        for c in shuffled:
            
            if max_score >= 0.99: # Near-perfect score, break early
                if self.verbose: 
                    print('Maxiumum performance reached')
                break
            
            if c in best_C0 and len(best_C0) > 1:
                C_0_prime = [elem for elem in best_C0 if elem != c]
                C_1_prime = best_C1 + [c]
                pass_str='cluster 0 to cluster 1'
            elif c in best_C1 and len(best_C1) > 1:
                C_0_prime = best_C0 + [c]
                C_1_prime = [elem for elem in best_C1 if elem != c]
                pass_str='cluster 1 to cluster 0'
            else:
                continue
            
            # Evaluate the new split
            score_temp = self._evaluate_split(super_classes=[C_0_prime, C_1_prime])
            if score_temp > max_score:
                max_score = score_temp
                best_C0, best_C1 = C_0_prime, C_1_prime
                                
                if self.verbose: 
                    print(f'{c} passed from {pass_str}')
                    
        return best_C0, best_C1, max_score

    def _lsoo(self, classes):
        """Leave-Salient-One-Out."""
        np.random.seed(self.random_state)
        shuffled = classes.copy()
        np.random.shuffle(shuffled)
        max_score = -1.0
        best_C0, best_C1 = None, None

        for c in shuffled:
            
            C_0_prime = [c]
            C_1_prime = [elem for elem in shuffled if elem != c]
            
            score = self._evaluate_split(super_classes=[C_0_prime, C_1_prime])
            if score > max_score:
                max_score = score
                best_C0, best_C1 = C_0_prime, C_1_prime
            
            if max_score >= 0.99:
                if self.verbose: 
                    print('Maxiumum performance reached')
                break
        
        return best_C0, best_C1, max_score
        
    def _split_function(self, node_classes):
        """Dispatch to the selected SSF."""
        if self.ssf == 'potr':
            return self._potr(node_classes)
        elif self.ssf == 'srtr':
            return self._srtr(node_classes)
        elif self.ssf == 'lsoo':
            return self._lsoo(node_classes)
        else:
            raise ValueError(f"Unknown SSF: {self.ssf}")

    def _generate_tree_hash(self, tree_structure):
        """Generate unique hash for a tree to check for duplicates."""
        def normalize_node(node):
            if isinstance(node, (list, np.ndarray)) and len(node) > 0:
                if isinstance(node[0], (int, str, np.integer)):  # Leaf node
                    return tuple(sorted(node))
                else:  # Parent node
                    return tuple(sorted(normalize_node(child) for child in node))
            return node
        
        normalized_tree = normalize_node(tree_structure)
        return hashlib.md5(str(normalized_tree).encode()).hexdigest()
    
    def reset_model(self):
        """Reset the model to initial state."""
        self.Z = np.zeros((self.n_clusters, 4))
        self.id_node = 2 * self.n_object - 2  # count down from all nodes
        self.id_next = 2 * self.n_object - 2
        self.id_clust = self.n_object - 1  # count down from all non-singleton clusters
        self.clusters = {self.id_node: list(range(self.n_object))}  # non-singleton clusters        
        self.is_fitted = False
        self.tree_hash = None
        return self
    
    def linkage2nodes(self, Z):
        """Convert linkage matrix to node hierarchy."""
        # Implementation depends on your HierNode structure
        nodes = []
        for i in range(Z.shape[0]):
            node = HierNode()
            node.set_id(i)
            node.set_branch_ids([int(Z[i, 0]), int(Z[i, 1])])
            node.set_height(Z[i, 2])
            nodes.append(node)
        return nodes
    
    def nodes2linkage(self, nodes):
        """Convert node hierarchy to linkage matrix."""
        Z = np.zeros((len(nodes), 4))
        for i, node in enumerate(nodes):
            Z[i, 0] = node.branch_ids[0]
            Z[i, 1] = node.branch_ids[1]
            Z[i, 2] = node.height
            Z[i, 3] = node.size
        return Z
    
    def fit(self, fit_nodes=True, sort_=True, refit_=False):
        """Fit the hierarchical clustering model."""
        if self.dset is None:
            raise ValueError("No dataset provided. Initialize with dset parameter.")
        
        if self.is_fitted:
            if refit_:
                warnings.warn('Model already fitted. Resetting for refit.')
                self.reset_model()
            else:
                raise ValueError('Model already fitted. Set refit_=True to refit.')
        
        # Initialize model state
        self.reset_model()
        
        if fit_nodes:
            nodes_ = [HierNode() for _ in range(self.n_clusters)]        
            for i in range(self.n_clusters):
                nodes_[i].set_id(i)
                nodes_[i].node_type = 'parent'
        
        Z_temp = self.Z.copy()
        
        for id_n, id_clust in enumerate(range(self.n_clusters-1, -1, -1)):
            current_cluster = self.clusters[self.id_next]
            height = len(current_cluster)
            
            if height > 2:
                # Use SSF to split the cluster
                child_0, child_1, split_score = self._split_function(current_cluster)
            else:
                # Base case: split the two elements
                child_0 = [current_cluster[0]]
                child_1 = [current_cluster[1]]                                      
                        
            # Sort if requested
            if sort_:
                child_0.sort()
                child_1.sort()
            
            # Assign IDs to children
            child_0_id = self._assign_cluster_id(child_0)
            child_1_id = self._assign_cluster_id(child_1)
            
            # Update linkage matrix
            Z_temp[id_clust, 0] = child_0_id
            Z_temp[id_clust, 1] = child_1_id
            Z_temp[id_clust, 2] = height
            Z_temp[id_clust, 3] = len(child_0) + len(child_1)
            
            # Update clusters dictionary
            if len(child_0) > 1:
                self.clusters[child_0_id] = child_0
            if len(child_1) > 1:
                self.clusters[child_1_id] = child_1
            
            self.id_next -= 1
            
            if fit_nodes and id_n < len(nodes_):
                nodes_[id_n].append_subsets(child_0)
                nodes_[id_n].append_subsets(child_1)
                nodes_[id_n].set_branch_ids([child_0_id, child_1_id])
                    
        Z_temp = monotonize_rescale_(Z_temp, swap=0)
        self.Z = Z_temp
        self.is_fitted = True
        
        # Generate tree hash for duplicate detection
        tree_structure = self.clusters if fit_nodes else Z_temp
        self.tree_hash = self._generate_tree_hash(tree_structure)
        
        if fit_nodes:
            self.nodes_ = nodes_
            return Z_temp, nodes_
        else:
            return Z_temp
    
    def _assign_cluster_id(self, cluster):
        """Assign ID to a cluster, creating new ID for non-singleton clusters."""
        if len(cluster) == 1:
            return cluster[0]  # Use original class ID for singletons
        else:
            self.id_node -= 1
            return self.id_node
    
    def get_tree_hash(self):
        """Get the unique hash of the current tree structure."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        return self.tree_hash