# built-in libraries
import numpy as np
import math
from itertools import combinations, islice
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
    
def sort_class_ids(y):
    classes=np.unique(y)
    Nclass = len(classes)
    # Standard order of class ids: 0,1,..,N-1
    if np.any(classes!=np.arange(Nclass)):
        for i in range(Nclass):
            y[y==classes[i]]=i         
        # classes=np.unique(y_train)
    return y

def _handle_small_dataset_fallback(x_train, y_train, original_n_va):
    """
    Automatically adjust n_va based on dataset size and class distribution
    """
    n_samples = len(y_train)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = min(class_counts)
    
    # Determine appropriate number of validation folds
    if n_samples < 20 or min_class_count < 4:
        # Very small dataset, use LOOCV or reduced folds
        if min_class_count == 1:
            # Some classes have only 1 sample, can't split
            return 1  # Single validation run
        elif min_class_count == 2:
            return 2  # 2-fold CV
        elif min_class_count == 3:
            return 3  # 3-fold CV
        else:
            return min(original_n_va, 3)  # Cap at 3 folds
    elif n_samples < 50 or min_class_count < 6:
        # Small dataset, reduce folds
        return min(original_n_va, 4)
    else:
        # Normal dataset, use original setting
        return original_n_va

def _split_data_with_fallback(X, y, shuffle_=True, 
                              min_samples_per_class=2, 
                              rseed=0, verbose=3):
    """
    Split data with fallback mechanism for small classes.
    Tries different split strategies if stratification fails.
    """
    fallback_strategies = [
        {'test_size': 0.25, 'stratify': True},  # Original: 4-fold (25%)
        {'test_size': 0.33, 'stratify': True},  # Fallback 1: 3-fold (33%)
        {'test_size': 0.50, 'stratify': True},  # Fallback 2: 2-fold (50%)
        {'test_size': 0.25, 'stratify': False}, # Fallback 3: No stratification
        {'test_size': 0.50, 'stratify': False}  # Fallback 4: No stratification, 50%
    ]
    
    for attempt, strategy in enumerate(fallback_strategies):
        try:
            if strategy['stratify']:
                # Check if we have enough samples for stratification
                unique_classes, class_counts = np.unique(y, return_counts=True)
                
                # Ensure each class has at least min_samples_per_class in both train and test
                min_required = min_samples_per_class * 2  # At least 2 in each split
                if any(count < min_required for count in class_counts):
                    if verbose>=3:
                        print(f"Class counts too small for stratification: {dict(zip(unique_classes, class_counts))}")
                    continue  # Try next strategy
                
                # Use stratified split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, 
                    test_size=strategy['test_size'],
                    stratify=y,
                    random_state=rseed + attempt,
                    shuffle=shuffle_
                )
            else:
                # Use simple random split without stratification
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=strategy['test_size'],
                    stratify=None,
                    random_state=rseed + attempt,
                    shuffle=shuffle_
                )
            
            # Verify the split is valid
            if _validate_split(y_train, y_val, min_samples_per_class, verbose):
                if attempt > 0 and verbose>=3:
                    print(f"Used fallback strategy {attempt}: test_size={strategy['test_size']}, "
                          f"stratify={strategy['stratify']}")
                return X_train, X_val, y_train, y_val
                
        except ValueError as e:
            if verbose>=3:
                print(f"Split attempt {attempt} failed: {e}")
            continue
    
    # If all strategies fail, use the last available samples
    if verbose>=3:
        print("All split strategies failed, using last samples as validation")
    split_idx = int(len(X) * 0.75)  # 75% train, 25% validation
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
           
def _validate_split(y_train, y_val, min_samples_per_class, verbose):
    """Validate that the split has sufficient samples in each class"""
    train_classes = np.unique(y_train)
    val_classes = np.unique(y_val)
    
    # Check if all classes are represented in both sets
    if not np.array_equal(train_classes, val_classes):
        if verbose>=3:
            print(f"Class mismatch: train={train_classes}, val={val_classes}")
        return False
    
    # Check minimum samples per class
    for y_split in [y_train, y_val]:
        unique, counts = np.unique(y_split, return_counts=True)
        if any(count < min_samples_per_class for count in counts):
            if verbose>=3:
                print(f"Insufficient samples: {dict(zip(unique, counts))}")
            return False
    
    return True

def sort_class_id(y):
    classes=np.unique(y)
    Nclass = len(classes)
    # Standard order of class ids: 0,1,..,N-1
    if np.any(classes!=np.arange(Nclass)):
        for i in range(Nclass):
            y[y==classes[i]]=i         
        # classes=np.unique(y_train)
    return y

def encode_super_labels(super_classes, y, x=None, reorder=True):
    sel_inds = np.array([],dtype=int)
    y_sel = y.copy()
    for i,g in enumerate(super_classes):
        # find where indices for selected memberships occur
        if (type(g) is list or type(g) is np.ndarray) and len(g)>1:
            for j in g:
                loc_ = np.where(y==j)[0].astype(int)
                sel_inds = np.r_[sel_inds,loc_]
                if reorder:
                    y_sel[y==j]=i
                else:
                    y_sel[y==j]=g[0]
        else:
            loc_ = np.where(y==g)[0].astype(int)
            sel_inds = np.r_[sel_inds,loc_]
            if reorder:
                y_sel[y==g]=i
            else:
                y_sel[y==g]=g
    y_sel = y_sel[sel_inds]
    if x is None:
        return y_sel
    else:        
        x_sel = x[sel_inds,:].copy()
        return y_sel, x_sel

def class_labels_sanity_check(y_train,y_test):
    classes_train = np.unique(y_train)
    classes_test = np.unique(y_test)
    if np.all(classes_train==classes_test):
        classes=classes_train
    else:
        raise ValueError("Discrepancy bw train and test labels. Class labels don't match.")
    return classes
   
def monotonize_rescale_(Z,swap=False):
    # Monotonize and rescale the tree linkage
    Zh = Z[:,2]
    n = len(Zh)
    
    start_ = Zh.max()/(n)
    stop_ = Zh.max()
    Zh[:] = np.round(np.linspace(start_,stop_,n),2)
    
    if np.ptp(Zh)/(n-1) < 0.05:
        scale_=0.05*(n-1)/np.ptp(Zh)
        Zh *= scale_  
    
    # Swap column 0 and 1 for compatiblity with dendogram
    if swap:
        z0=Z[:,0].copy()
        Z[:,0]=Z[:,1]
        Z[:,1]=z0
    
    return Z

def plot_dendrogram(Z, close_all=0, orient="top", leafFont=9, title_=False, class_list=None):
    if close_all:plt.close('all')
    if title_: title_text= "Hierarchical Clustering Dendrogram"
    if class_list is not None:
        sch.dendrogram(Z,orientation=orient,leaf_font_size=leafFont,labels=[txt for txt in class_list])
    else:
        sch.dendrogram(Z,orientation=orient,leaf_font_size=leafFont)
    if title_: plt.title(title_text)

def generate_rand_dist(n_dim,type_='diss'):
    np.random.seed(seed=None)    
    d = np.zeros((n_dim,n_dim))
    
    a = 1
    b = 0
    if type_ == 'perturb':
        a = 2
        b = 0.5
    
    for i in combinations(range(n_dim),2):        
        rr=np.round(a*(np.random.rand()-b),2)
        d[i]=rr
        d[i[::-1]]=rr
        
    return d

def nCk(n,k):
    f = math.factorial
    return f(n) // f(k) // f(n-k)

def batched(iterable, n):
      "Batch data into lists of length n. The last batch may be shorter."
      # batched('ABCDEFG', 3) --> ABC DEF G
      if n < 1:
          raise ValueError('n must be >= 1')
      it = iter(iterable)
      while (batch := list(islice(it, n))):
          yield batch

def C_n(n):
    """All divisions of an n-element cluster into two non-empty 
       subsets: 2**(n-1)-1
    """
    sum_=0
    for k in range(1,round(n/2)+1):
        if n%2==0 and k==n/2:
            sum_ += int(nCk(n,k)/2)
        else:
            sum_ += nCk(n,k)
    print(sum_)
    return sum_

def T_n(n):
    "Estimate total number of trees given number of classes"    
    if n==2:        
        return 1
    elif n==3:
        return 3
    elif n>3:
        sum_=0
        for i in range(1,round(n/2)+1):
            if n%2==0 and i==n/2:
                sum_ += int(nCk(n,n-i)/2)*T_n(n-i)
            else:
                sum_ += nCk(n,n-i)*T_n(n-i)
        return sum_

def T_n_look(n):
    "Estimate total number of trees given number of classes"
    table_=[0,0]
    if n==2:
        table_.append(1)
        return table_
    elif n==3:
        table_.append(1)
        table_.append(3)
        return table_
    elif n>3:
        table_.append(1)
        table_.append(3)
        for n_i in range(4,n+1):
            sum_=0           
            for k in range(1,round(n_i/2)+1):
                if n_i%2==0 and k==n_i/2:
                    sum_ += int(nCk(n_i,n_i-k)/2)*table_[n_i-k]                    
                else:
                    sum_ += nCk(n_i,n_i-k)*table_[n_i-k]
            table_.append(sum_)
        return table_
    
def compare_(root1, root2, sol):
    if root1 is not None and root2 is not None:
        if root1.subsets[0]==root2.subsets[0]:
            sol.append(1)
            compare_(root1.left, root2.left, sol)
            compare_(root1.right, root2.right, sol)
        elif root1.subsets[0]==root2.subsets[1]:
            sol.append(2)
            compare_(root1.left, root2.right, sol)
            compare_(root1.right, root2.left, sol)
        else:            
            sol.append(0)
    else:
        sol.append(-1)

def isEqual(root1, root2):
    eq_=[]
    compare_(root1,root2,eq_)
    if 0 in eq_:
        return False
    else:
        return True

def compare_tree(n_i,Yi,n_j,Yj):
    if np.all(Yi==Yj):
        return True
    elif isEqual(n_i[0],n_j[0]):
        return True
    else:
        return False

def preprocess_diss_mat(D,scale_=True):
    # print(D)
    D[np.isnan(D)]=np.nanmax(D)
    m=D.shape[0]
    inds_i=[i for i in range(m) for j in range(m) if i!=j]
    inds_j=[j for i in range(m) for j in range(m) if i!=j]
    if len(D[inds_i,inds_j][D[inds_i,inds_j]==0])>0:
        D[inds_i,inds_j]=D[inds_i,inds_j]+(D[inds_i,inds_j][D[inds_i,inds_j]!=0]).min()
    if scale_: D = (D - np.min(D)) / (np.max(D) - np.min(D))
    return D

def jensen_shannon_dist(x1,x2):
    pdf1, bin_edges = np.histogram(x1, bins=50, density=True)
    pdfn1 = pdf1/pdf1.sum()
    
    pdf2, bin_edges = np.histogram(x2, bins=50, density=True)
    pdfn2 = pdf2/pdf2.sum()
    
    return distance.jensenshannon(pdfn1, pdfn2, 2)

def to_one_hot(y_pred, num_classes=None):
    y_pred = np.asarray(y_pred)
    if num_classes is None:
        num_classes = np.max(y_pred) + 1
    one_hot = np.zeros((len(y_pred), num_classes))
    one_hot[np.arange(len(y_pred)), y_pred] = 1
    return one_hot
