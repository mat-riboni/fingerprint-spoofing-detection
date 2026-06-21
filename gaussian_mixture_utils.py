from gaussian_utils import *
from utils import *

from utils import *
import scipy
import json
import numpy

def logpdf_GMM(X, gmm):
    """
    gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]
    """

    S = numpy.zeros((len(gmm), X.shape[1]))

    for i, (w, mu, C) in enumerate(gmm):
        S[i, :] = logpdf_GAU_ND(X, mu, C) + np.log(w)
    logdens = scipy.special.logsumexp(S, axis=0)
    return S, logdens

def EM(X, gmm_0, threshold=1e-6, psi=0.01):
    actual_gmm = gmm_0
    n_components = len(gmm_0)
    D = X.shape[0]
    N = X.shape[1]
    prev_ll = None

    while True:
        joint_log, logdens = logpdf_GMM(X, actual_gmm)

        avg_ll = np.sum(logdens) / N
        if prev_ll is not None and avg_ll - prev_ll < threshold:
            break
        prev_ll = avg_ll

        resps = np.exp(joint_log - logdens)

        Z = np.sum(resps, axis=1)
        F = np.zeros((n_components, D))
        S_stat = np.zeros((n_components, D, D))

        for g in range(n_components):
            F[g, :] = np.sum(resps[g, :] * X, axis=1)
            S_stat[g, :, :] = (X * resps[g, :]) @ X.T

        next_mu    = F / Z[:, np.newaxis]
        next_sigma = (S_stat / Z[:, np.newaxis, np.newaxis]
                      - next_mu[:, :, np.newaxis] * next_mu[:, np.newaxis, :])
        next_w     = Z / N

        for g in range(n_components):
            U, s, _ = np.linalg.svd(next_sigma[g])
            s[s < psi] = psi
            next_sigma[g] = U @ (s[:, np.newaxis] * U.T)

        actual_gmm = [(next_w[g], next_mu[g].reshape(D, 1), next_sigma[g])
                      for g in range(n_components)]

    return actual_gmm
        

def LBG(X, n_components, alpha = 0.1):
    mu0 = vcol(np.sum(X, axis=1)) / X.shape[1]
    C0 = np.cov(X, bias=True)
    GMM = [(1.0, mu0, C0)]
    while(len(GMM) < n_components):
        temp_GMM = []
        for (w, mu, Sigma) in GMM:
            U, s, Vh = numpy.linalg.svd(Sigma)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            temp_GMM.extend([(w/2, mu - d, Sigma), (w/2, mu + d, Sigma)])
        GMM = temp_GMM
        GMM = EM(X, GMM)
        _, logdens = logpdf_GMM(X, GMM)
    return GMM


def compute_min_dcf(scores, L, pi, C_fn, C_fp):
    thresholds = np.sort(np.hstack([-np.inf, scores, np.inf]))

    rscores = vrow(scores)
    cthresholds = vcol(thresholds)
    preds_mat = (rscores > cthresholds) * 1

    actual_1 = (L == 1)
    actual_0 = (L == 0)

    P_fp = np.sum((preds_mat == 1) & (actual_0), axis=1) / np.sum(actual_0)
    P_fn = np.sum((preds_mat == 0) & (actual_1), axis=1) / np.sum(actual_1)
    
    B_all = (pi * C_fn * P_fn) + ((1 - pi) * C_fp * P_fp)
    B1 = pi*C_fn
    B2 = (1 - pi)*C_fp
    B_dummy = min(B1, B2)
    
    return np.min(B_all) / B_dummy

def compute_act_dcf(preds, L, pi, C_fn, C_fp):
    cf = binary_confusion_matrix(preds, L) 
    P_fn = cf[0][1] / (cf[0][1] + cf[1][1])
    P_fp = cf[1][0] / (cf[1][0] + cf[0][0])
    B = (pi*C_fn*P_fn) + ((1 - pi)*C_fp*P_fp)
    B1 = pi*C_fn
    B2 = (1 - pi)*C_fp
    B_dummy = min(B1, B2)
    return B / B_dummy
