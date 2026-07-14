import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

def compute_categories(X, y, neighbors_arr):
    categories = []
    for i in range(len(X)):
        if y[i] == 0:
            categories.append("clean")
        else:
            neigh_labels = y[neighbors_arr[i]]
            n_bug = sum(lbl > 0 for lbl in neigh_labels)
            if n_bug == 0:
                categories.append("isolated")
            elif n_bug <= 2:
                categories.append("sparse")
            else:
                categories.append("dense")
    return categories

def smote_sparse(X, y, categories, neighbors_arr, n_samples=200):
    synthetic_X, synthetic_y = [], []
    sparse_indices = [i for i, cat in enumerate(categories) if cat == "sparse"]
    if not sparse_indices:
        return np.empty((0, X.shape[1])), np.array([])
    for _ in range(n_samples):
        i  = random.choice(sparse_indices)
        xi = X[i]
        neigh_bug = [j for j in neighbors_arr[i] if y[j] > 0]
        if not neigh_bug:
            continue
        j   = random.choice(neigh_bug)
        lam = random.random()
        synthetic_X.append(xi + lam * (X[j] - xi))
        synthetic_y.append(y[i])
    return np.array(synthetic_X), np.array(synthetic_y)

def penn_undersample(X, y, neighbors_arr, beta=0.5):
    keep = []
    for i in range(len(X)):
        if y[i] > 0:
            keep.append(i)
        else:
            clean_ratio = np.mean(y[neighbors_arr[i]] == 0)
            if clean_ratio >= beta:
                keep.append(i)
    return keep

def smote_penn_balance(X, y, file_names, k=5, beta=0.5, n_smote=200):
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    neighbors_arr = nn.kneighbors(X, return_distance=False)
    categories = compute_categories(X, y, neighbors_arr)
 
    keep_idx = penn_undersample(X, y, neighbors_arr, beta=beta)
    X_p = X[keep_idx]
    y_p = y[keep_idx]
    names_p = [file_names[i] for i in keep_idx]
    cats_p  = [categories[i] for i in keep_idx]
 
    nn2 = NearestNeighbors(n_neighbors=k).fit(X_p)
    neighbors_p = nn2.kneighbors(X_p, return_distance=False)
 
    syn_X, syn_y = smote_sparse(X_p, y_p, cats_p, neighbors_p, n_samples=n_smote)
 
    if len(syn_X) > 0:
        X_bal   = np.vstack([X_p, syn_X])
        y_bal   = np.concatenate([y_p, syn_y])
        names_bal = names_p + ['synthetic'] * len(syn_y)
    else:
        X_bal, y_bal, names_bal = X_p, y_p, names_p
 
    return X_bal, y_bal, names_bal


def extract_tabular_metrics(code_str):
    features = np.zeros(20)
    
    lines = code_str.split('\n')
    loc = len([line for line in lines if line.strip() != ""])
    features[10] = loc
    
    wmc = code_str.count("public") + code_str.count("private") + code_str.count("protected")
    features[0] = max(1, wmc)
    
    cc = code_str.count("if (") + code_str.count("for (") + code_str.count("while (") + code_str.count("case ")
    features[18] = max(1, cc)
    
    return features.reshape(1, -1)