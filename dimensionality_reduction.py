import numpy as np

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