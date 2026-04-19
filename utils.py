import numpy
import matplotlib.pyplot as plt


def load(file_path):
    D = []
    L = []
    with open(file_path) as file:
        for line in file:
            fields = line.split(',')
            feature = []
            for i in range(6):
                feature.append(float(fields[i].strip()))
            L.append(float(fields[6].strip()))
            D.append(numpy.array(feature))
    L = numpy.array(L)
    D = numpy.array(D)
    return D.T, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)


def plot_features(D, L):
    feature_names = [f'Feature {i+1}' for i in range(6)]
    class_names = {0: 'Fake', 1: 'Genuine'}
    colors = ['red', 'blue']

    fig, axes = plt.subplots(6, 6, figsize=(20, 20), constrained_layout=True)

    for i in range(6):     
        for j in range(6):  
            ax = axes[i, j]
            
            if i == j:
                for label in [0, 1]:
                    x_class = D[i, L == label]
                    ax.hist(x_class, bins=30, alpha=0.5, density=True, 
                            label=class_names[label], color=colors[label])
                ax.set_title(f"Hist {feature_names[i]}")
                ax.set_ylabel("Density")
            else:
                for label in [0, 1]:
                    x_data = D[j, L == label]
                    y_data = D[i, L == label] 
                    ax.scatter(x_data, y_data, alpha=0.3, s=5, 
                            color=colors[label], label=class_names[label])
            
            if i == 5: ax.set_xlabel(feature_names[j])
            if j == 0: ax.set_ylabel(feature_names[i])
            
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize='small')

    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(D, L, title_prefix="PCA Direction"):
    num_dims = D.shape[0]
    labels = np.unique(L)
    
    cols = 3
    rows = (num_dims + cols - 1) // cols
    
    plt.rcParams["figure.figsize"] = (15, 5 * rows)
    
    for i in range(num_dims):
        plt.subplot(rows, cols, i + 1)
        
        for label in labels:
            plt.hist(D[i, L == label], bins=25, alpha=0.5, label=f"Class {label}", density=True)
        
        plt.title(f"{title_prefix} {i + 1}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
    
    plt.tight_layout()
    plt.show() 

def vrow(v):
    return v.reshape(1, v.size)

def vcol(v):
    return v.reshape(v.size, 1)


def print_corr(Corr, title="Correlation Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(Corr, cmap='coolwarm', vmin=-1, vmax=1)

    fig.colorbar(cax)

    for i in range(Corr.shape[0]):
        for j in range(Corr.shape[1]):
            text_color = "white" if abs(Corr[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{Corr[i, j]:.2f}", 
                    ha="center", va="center", color=text_color, fontsize=10)

    ticks = np.arange(Corr.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"Feat {i+1}" for i in ticks])
    ax.set_yticklabels([f"Feat {i+1}" for i in ticks])

    plt.title(title)
    plt.tight_layout()
    plt.show()