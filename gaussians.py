import numpy as np

def logpdf_GAU_ND(X : np.ndarray , mu : np.ndarray, C : np.ndarray):
    """
    Computes logaritmic multivariate gaussian density for samples X
    """
    M = X.shape[0]
    C_inv = np.linalg.inv(C)
    _, log_C = np.linalg.slogdet(C)
    X_centered = X - mu
    P = C_inv @ X_centered
    mahalanobis_sqrd = np.sum((X_centered * P), axis=0)
    Y = - ((M* np.log(2 * np.pi)) / 2) - (log_C / 2) - (mahalanobis_sqrd / 2)
    return Y

def loglikelihood(X, mu, C):
    Y = logpdf_GAU_ND(X, mu, C)
    return np.sum(Y, axis=0)
