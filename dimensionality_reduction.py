import numpy as np
import scipy

def PCA(D: np.ndarray, m: int) -> np.ndarray:
    """
    1. Center the input data
    2. Computes covariance matrix C
    3. Find eigen values/vectors
    4. Builds P using the m largest eigen values and their corresponding eigen vectors 
    Args:
        D: input data (the dataset)
        m: new number of dimensions, has to be lower than D.shape[0]
    Returns:
        P: new projection matrix that will be used to project the dataset in m dimensions
    """
    mu = D.mean(axis=1)
    mu = mu.reshape(mu.size, 1)
    DC = D - mu
    C  = (1/DC.shape[1])*(DC @ DC.T)
    s, U = np.linalg.eigh(C)
    P = U[: , ::-1][: , :m]
    return P

def LDA(D: np.ndarray, L: np.ndarray, m: int) -> np.ndarray:
    """
    Args:
        D: input data
        L: labels for input data
        m: number of dimensions (at most C - 1, where C is the number of classes)
    Returns:
        U: new projection matrix
    """
    SW = np.zeros((D.shape[0] ,D.shape[0]))
    SB = np.zeros((D.shape[0], D.shape[0]))
    labels = np.unique(L)
    mu = D.mean(axis=1)
    mu = mu.reshape(mu.size, 1)
    for i in range(labels.size):
        Di =  D[:, labels[i]==L]
        mui = Di.mean(axis=1)
        mui = mui.reshape(mui.size, 1)
        Di = Di - mui
        SWi = Di @ Di.T
        SW = SW + SWi
        SBi = Di.shape[1] * ((mui - mu)@(mui - mu).T)
        SB = SB + SBi
    SW = SW/D.shape[1]
    SB = SB/D.shape[1]

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, :m]
    return W