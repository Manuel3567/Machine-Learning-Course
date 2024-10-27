import numpy as np

def pca(X, n_components):

    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    X_std = np.std(X_centered, axis=0, ddof=1)
    X_norm = X_centered / X_std
    
    U, Sigma, Vt = np.linalg.svd(X_norm, full_matrices=False)
    
    principal_components = Vt.T[:, :n_components]
    
    #projections = X_norm @ principal_components #siehe Wikipedia U @ Sigma = X @ V, da V transponiert @ V = Einheitsmatrix = 1
    projections = U @ np.diag(Sigma[:n_components])

    n = X.shape[0]
    
    std_devs = Sigma[:n_components] / np.sqrt(n - 1) # Der Faktor (n - 1) wird von SVD nicht angewandt und muss von Hand berücksichtigt werden
    # Die Elemente von SVD Sigma ist Wurzel (n - 1) * dem tatsächlichen Sigma
    #std_devs = Sigma[:n_components]

    
    return principal_components, projections, std_devs
