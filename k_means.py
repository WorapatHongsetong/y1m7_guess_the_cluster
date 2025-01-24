import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

DATA = genfromtxt("data/points.csv", delimiter=",")

def get_boarder(data: np.array) -> tuple:
    x_min = np.min(data[:, 0]) 
    y_min = np.min(data[:, 1])
    x_max = np.max(data[:, 0])
    y_max = np.max(data[:, 1])
    return x_min, y_min, x_max, y_max

# Yanked from homework.
def k_means(data, k, max_iters=100, error=1e-4):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    old_centroids = np.zeros_like(centroids)
    
    cluster_assignments = np.zeros(data.shape[0])
    
    for _ in range(max_iters):
        for i, point in enumerate(data):
            distances = np.linalg.norm(point - centroids, axis=1)
            cluster_assignments[i] = np.argmin(distances)
        
        old_centroids = centroids.copy()
        
        for i in range(k):
            assigned_points = data[cluster_assignments == i]
            if len(assigned_points) > 0:
                centroids[i] = np.mean(assigned_points, axis=0)
        
        if np.all(np.abs(centroids - old_centroids) < error):
            break

    return centroids, cluster_assignments








if __name__ == "__main__":

    DATA = genfromtxt("data/points.csv", delimiter=",")

    x_min, y_min, x_max, y_max = get_boarder(DATA)

    centroids, cluster_assign = k_means(data=DATA, k=4)
    print(centroids)

    plt.scatter(DATA[:, 0], DATA[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1])
    plt.plot((x_min, x_max), (y_min, y_min))
    plt.plot((x_min, x_max), (y_max, y_max))
    plt.plot((x_max, x_max), (y_min, y_max))
    plt.plot((x_min, x_min), (y_min, y_max))
    plt.show()