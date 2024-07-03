import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin
from PIL import Image
from skimage import io

def computeCentroid(image, indices):
    pixels = [image[idx] for idx in indices]

    # Computing the mean of RGB values
    centroid = np.mean(pixels, axis=0)

    return centroid

def mykmeans(X, k, max_iters=100):
    #Initializing cluster centers randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        #Assigning each data point to the nearest cluster center
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        #Updating cluster centers
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Checking for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids

def compress_image(image, centroids):
    # Reshaping the image into a 2D array of pixels
    pixels = np.reshape(image, (-1, 3))
    print("Shape of pixels array:", pixels.shape)

    # Calculating distances from each pixel to each centroid
    distances = np.linalg.norm(pixels[:, np.newaxis, :] - centroids, axis=2)

    # Assigning each pixel to the nearest centroid
    labels = np.argmin(distances, axis=1)

    # Replacing each pixel with the color of its nearest centroid
    compressed_pixels = centroids[labels]

    # Reshaping the compressed pixels back into the original image shape
    compressed_image = np.reshape(compressed_pixels, image.shape)

    return compressed_image

def save_compressed_images(image, k_values, save_path):
    os.makedirs(save_path, exist_ok=True)

    original_path = os.path.join(save_path, 'original.png')
    plt.imsave(original_path, image)

    for k in k_values:
        # Performing K-means clustering
        centroids = mykmeans(image.reshape(-1, 3), k)

        # Compressing the image using centroids
        compressed_image = compress_image(image, centroids)

        compressed_path = os.path.join(save_path, f'compressed_K{k}.png')
        plt.imsave(compressed_path, compressed_image)

image = plt.imread('/content/drive/MyDrive/ML Data/test.png')
k_values = [3, 4, 6, 8]
save_path = '/content/compressed_images/'

save_compressed_images(image, k_values, save_path)


def compressed_images(image, k_values, save_path):
    os.makedirs(save_path, exist_ok=True)

    original_path = os.path.join(save_path, 'original.png')
    plt.imsave(original_path, image)

    for k in k_values:
        # Performing K-means clustering using scikit-learn's KMeans
        kmeans = KMeans(n_clusters=k, random_state=0).fit(image.reshape(-1, 3))
        centroids = kmeans.cluster_centers_

        # Compressing the image using centroids
        compressed_image = compress_image(image, centroids)

        compressed_path = os.path.join(save_path, f'compressed_K{k}_sklearn.png')
        plt.imsave(compressed_path, compressed_image)



image = plt.imread('/content/drive/MyDrive/ML Data/test.png')
k_values = [3, 4, 6, 8]
save_path = '/content/compressed_images_Kmeans/'

compressed_images(image, k_values, save_path)


def compute_spatial_distance(pixel1, pixel2):
    spatial_dist = np.linalg.norm(pixel1 - pixel2)
    return spatial_dist

def mykmeans_spatial(X, k, max_iters=100, spatial_weight=0.5):
    #Initializing cluster centers randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Assigning each data point to the nearest cluster center
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        
        # Calculate spatial distances
        spatial_distances = np.array([[compute_spatial_distance(X[i], X[j]) for j in range(X.shape[0])] for i in range(X.shape[0])])
        
        # Combine color distance and spatial distance with a weighted sum
        combined_distances = (1 - spatial_weight) * distances + spatial_weight * spatial_distances
        
        labels = np.argmin(combined_distances, axis=1)

        # Updating cluster centers
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Checking for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids

def save_compressed_images_spatial(image, k_values, save_path):
    # Create a directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the original image
    original_path = os.path.join(save_path, 'original.png')
    plt.imsave(original_path, image)

    for k in k_values:
        # Performing K-means clustering with spatial coherence
        centroids = mykmeans_spatial(image.reshape(-1, 3), k)

        # Compressing the image using centroids
        compressed_image = compress_image(image, centroids)

        # Save the compressed image
        compressed_path = os.path.join(save_path, f'compressed_K{k}_spatial.png')
        plt.imsave(compressed_path, compressed_image)

image = plt.imread('/content/drive/MyDrive/ML Data/test.png')
k_values = [3, 4, 6, 8]
save_path = '/content/compressed_images_spatial/'

save_compressed_images_spatial(image, k_values, save_path)