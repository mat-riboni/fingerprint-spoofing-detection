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