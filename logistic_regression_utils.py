from utils import vcol, vrow
from gaussian_utils import compute_min_dcf, compute_act_dcf
import numpy as np
import scipy


def logreg_obj(v, DTR, LTR, l):
    w = v[0:-1]
    w = vcol(w)
    b = v[-1]
    Z = 2*LTR - 1
    norm_sqrd = (w.T @ w).item()
    J = (l/2)*norm_sqrd + (1/LTR.shape[0]) * np.sum(np.logaddexp(0,-vrow(Z) * (w.T @ DTR + b)))

    G = -Z / (1 + np.exp((vrow(Z)*(w.T @ DTR + b))))
    J_dw = (l*w).ravel() + (1/LTR.shape[0])*np.sum(G * DTR, axis=1)
    J_db = (1/LTR.shape[0])*np.sum(G)
    J_grad = np.append(J_dw, J_db)
    return (J, J_grad)
    

def logreg_obj_weighted(v, DTR, LTR, l, pi_target):
    w = v[0:-1]
    w = vcol(w)
    b = v[-1]
    Z = 2*LTR - 1
    norm_sqrd = (w.T @ w).item()
    nT = (LTR == 1).sum()
    nF = (LTR == 0).sum()
    XI = np.zeros(LTR.shape)
    XI[LTR == 1] = pi_target / nT 
    XI[LTR == 0] = (1 - pi_target) / nF 

    J = (l/2)*norm_sqrd + np.sum( XI * (np.logaddexp(0,-vrow(Z) * (w.T @ DTR + b))))

    G = -Z / (1 + np.exp((vrow(Z)*(w.T @ DTR + b))))
    J_dw = (l*w).ravel() + np.sum(XI * G * DTR, axis=1)
    J_db = np.sum(XI * G)
    J_grad = np.append(J_dw, J_db)
    return (J, J_grad)

def train_log_model(DTR, LTR, logreg_obj, l):
    x0 = np.zeros(DTR.shape[0] + 1)
    pi_emp = (LTR == 1).mean()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=x0, args=(DTR, LTR, l))
    return x, pi_emp


def evaluate_log_model(x, DVAL, LVAL, training_prior, application_prior):
    S = (vcol(x[0:-1]).T @ DVAL + x[-1]).ravel() #scores vector
    preds = (S>0).astype(int)
    accuracy = (preds == LVAL).mean()
    print(f"Error rate: {1 - accuracy}")
    llr = S - np.log((training_prior/(1-training_prior)))
    minDCF = compute_min_dcf(llr=llr, L=LVAL, pi=application_prior, C_fn=1, C_fp=1)
    actDCF = compute_act_dcf(llr=llr, L=LVAL, pi=application_prior, C_fn=1, C_fp=1)
    print(f"Min DCF: {minDCF}")
    print(f"Act DCF: {actDCF}")
    print('-------')
        

