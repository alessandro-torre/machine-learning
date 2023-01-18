import numpy as np
import sklearn.manifold  # for TSNE
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # for Axes3D


def main():
    
    #create gaussian clouds in 3D
    centers = np.array([
            [ 1,  1,  1],
            [ 1,  1, -1],
            [ 1, -1,  1],
            [ 1, -1, -1],
            [-1,  1,  1],
            [-1,  1, -1],
            [-1, -1,  1],
            [-1, -1, -1]
    ])
    data = []
    labels = list()
    n_per_clouds = 100
    for i,c in enumerate(centers):
        data.append(c + np.random.randn(n_per_clouds, 1))
        labels += [i]*n_per_clouds
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    
    # Plot the clouds in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=labels)
    plt.show()
    
    # perform dimensionality reduction
    tsne = sklearn.manifold.TSNE()
    data_ = tsne.fit_transform(data)
    
    # Plot the clouds in 2-D
    plt.scatter(data_[:,0], data_[:,1], c=labels)
    plt.show()