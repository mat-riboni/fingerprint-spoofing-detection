import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from utils import *


def L_obj(alpha, H_hat):
    n = H_hat.shape[0]
    ones = np.ones(n)
    L = 0.5 * (alpha @ H_hat @ alpha) - alpha @ ones
    L_grad = H_hat @ alpha - ones
    return L, L_grad

def train_linear_SVM(DTR, LTR, C=0.1, K=1.0):
    Z = (LTR * 2) - 1
    D_hat = np.vstack((DTR, K * np.ones((1, DTR.shape[1]))))
    G_hat = D_hat.T @ D_hat
    H_hat = np.outer(Z, Z) * G_hat

    alpha0 = np.zeros(DTR.shape[1])  

    alpha_star, duality_val, info = fmin_l_bfgs_b(
        func=L_obj,
        x0=alpha0,
        args=(H_hat,),      
        bounds=[(0, C)] * DTR.shape[1],
        factr=np.nan,
        pgtol=1e-5,
        maxfun=50000,
        maxiter=50000
    )

    w_hat_star = np.sum(alpha_star * Z * D_hat, axis=1)
    return w_hat_star

def linear_SVM_scores(DVAL, w_hat_star, K=1.0):
    DVAL_hat = np.vstack((DVAL, K * np.ones((1, DVAL.shape[1]))))
    S = np.sum(vcol(w_hat_star) * DVAL_hat, axis = 0)
    return S

def eval_SVM(S, LVAL):
    preds = (S > 0).astype(int) 
    accuracy = np.sum(preds == LVAL)/LVAL.shape[0]  
    return preds, accuracy


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

def train_polynomial_kernel_SVM(DTR, LTR, C=1.0, c=1, d=2, xi=1):
    Z = (LTR * 2) - 1
    poly_kern = (DTR.T @ DTR + c)**d + xi 
    H = np.outer(Z, Z) * poly_kern

    alpha0 = np.zeros(DTR.shape[1])  
    alpha_star, duality_val, info = fmin_l_bfgs_b(
        func=L_obj,
        x0=alpha0,
        args=(H,),
        bounds=[(0, C)] * DTR.shape[1],
        factr=np.nan,
        pgtol=1e-5,
        maxfun=50000,
        maxiter=50000
    )
    return alpha_star


def polynomial_kernel_SVM_scores(DTR, LTR, DVAL, alpha_star, c=1, d=2, xi=1):
    Z = (LTR * 2) - 1
    poly_kern_eval = (DTR.T @ DVAL + c)**d + xi
    S = np.sum(vcol(alpha_star * Z) * poly_kern_eval, axis=0)   
    return S

def squared_distances(A, B):
    a_norm_sqrd = np.sum(A * A, axis = 0)
    b_norm_sqrd = np.sum(B * B, axis = 0)
    term2ab = 2 * (A.T @ B)
    sqrd_dist = vcol(a_norm_sqrd) - term2ab + vrow(b_norm_sqrd)
    return sqrd_dist

def train_rbf_kernel_SVM(DTR, LTR, gamma, C=1.0, xi = 1):
        Z = (LTR * 2) - 1
        rbf_kern = np.exp(-gamma * (squared_distances(DTR, DTR))) + xi
        H = np.outer(Z, Z) * rbf_kern
        alpha0 = np.zeros(DTR.shape[1])  
        alpha_star, duality_val, info = fmin_l_bfgs_b(
            func=L_obj,
            x0=alpha0,
            args=(H,),
            bounds=[(0, C)] * DTR.shape[1],
            factr=np.nan,
            pgtol=1e-5,
            maxfun=50000,
            maxiter=50000
        )
        return alpha_star

def rbf_kernel_SVM_scores(DTR, LTR, DVAL, alpha_star, gamma, xi=1):
    Z = (LTR * 2) - 1
    rbf_kern_eval = np.exp(-gamma * (squared_distances(DTR, DVAL))) + xi
    S = np.sum(vcol(alpha_star * Z) * rbf_kern_eval, axis=0)   
    return S