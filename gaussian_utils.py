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

def loglikelihood_eval(X, mu, C):
    return logpdf_GAU_ND(X, mu, C)

def compute_ll_per_class(D, parameters):
    S = np.zeros((len(parameters), D.shape[1])) #likelihood scores
    for i, (mu, cov_mat) in enumerate(parameters):
        row_i = loglikelihood_eval(D, mu, cov_mat)
        S[i, :] = row_i
    return S


def compute_theta_parameters_per_class(D, L):
    num_classes = len(np.unique(L))
    theta = []
    for i in range(num_classes):
        D_clss = D[:, L==i]
        cov_mat = np.cov(D_clss, bias=True)
        mu = np.mean(D_clss, axis=1)
        mu = mu.reshape(mu.shape[0], 1)
        theta.append((mu, cov_mat))
    return theta
