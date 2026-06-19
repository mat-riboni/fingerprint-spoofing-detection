import numpy as np
from utils import *
import matplotlib.pyplot as plt


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



def print_binary_confusion_matrix(predictions, actual):
    num_classes = len(np.unique(actual))
    for i in range(num_classes):
        if i == 0:
            print(f"               Class {i}         ", end="")
        else:
            print(f"Class {i}         ", end="")

    print("\n")
    cf = np.zeros((num_classes , num_classes), dtype=int)
    for i in range(num_classes):
        print(f"Class {i}        ", end="")
        for j in range(num_classes):
            cf[i][j] = np.sum((actual==j) & (predictions==i))
            num_digits = len(str(abs(cf[i][j])))
            num_spaces = 16 - num_digits
            space = " " * num_spaces
            print(cf[i][j], end=space)
        print("\n")

def compute_optimal_bayes(llr, pi, C_fn, C_fp):
    threshold = -np.log( (pi*C_fn) /( (1-pi) * C_fp ))
    pred = (llr > threshold) * 1
    return pred 

def compute_min_dcf(llr, L, pi, C_fn, C_fp):
    thresholds = np.sort(np.hstack([-np.inf, llr, np.inf]))

    rllr = vrow(llr)
    cthresholds = vcol(thresholds)
    preds_mat = (rllr > cthresholds) * 1

    actual_1 = (L == 1)
    actual_0 = (L == 0)

    P_fp = np.sum((preds_mat == 1) & (actual_0), axis=1) / np.sum(actual_0)
    P_fn = np.sum((preds_mat == 0) & (actual_1), axis=1) / np.sum(actual_1)
    
    B_all = (pi * C_fn * P_fn) + ((1 - pi) * C_fp * P_fp)
    B1 = pi*C_fn
    B2 = (1 - pi)*C_fp
    B_dummy = min(B1, B2)
    
    return np.min(B_all) / B_dummy

def compute_act_dcf(llr, L, pi, C_fn, C_fp):
    P = compute_optimal_bayes(llr, pi, C_fn, C_fp)
    cf = binary_confusion_matrix(P, L) 
    P_fn = cf[0][1] / (cf[0][1] + cf[1][1])
    P_fp = cf[1][0] / (cf[1][0] + cf[0][0])
    B = (pi*C_fn*P_fn) + ((1 - pi)*C_fp*P_fp)
    B1 = pi*C_fn
    B2 = (1 - pi)*C_fp
    B_dummy = min(B1, B2)
    return B / B_dummy

def plot_bayes_error(llr, L, label="Model", color_act='r', color_min='b'):
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPrioOdds = 1 / (1 + np.exp(-effPriorLogOdds))
    
    act_dcf = [compute_act_dcf(llr, L, p, 1, 1) for p in effPrioOdds]
    min_dcf = [compute_min_dcf(llr, L, p, 1, 1) for p in effPrioOdds]
    
    plt.plot(effPriorLogOdds, act_dcf, label=f'actDCF ({label})', color=color_act)
    
    plt.plot(effPriorLogOdds, min_dcf, label=f'minDCF ({label})', color=color_min, linestyle='--')
    
    # Formatting
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel(r'prior log-odds $\log \frac{\tilde{\pi}}{1-\tilde{\pi}}$')
    plt.ylabel('DCF')
    plt.title('Bayes Error Plot')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
