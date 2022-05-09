import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)
from math import sqrt


def add_noise(data):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :return: data + noise, where noise~N(0,0.01^2)
    """
    noise = np.random.normal(loc=0, scale=0.01, size=data.shape)
    return data + noise


def visualize_results(data, labels, centroids, path):
    """
    Visualizing results of the kmeans model, and saving the figure.
    :param data: data as numpy array of shape (n, 2)
    :param labels: the final labels of kmeans, as numpy array of size n
    :param centroids: the final centroids of kmeans, as numpy array of shape (k, 2)
    :param path: path to save the figure to.
    """
    pass
    # plt.savefig(path)


def sum_scaling(values):
    min_val = min(values)
    sum_values = sum(values)
    for i in range(len(values)):
        values[i] = (values[i]-min_val)/sum_values


def transform_data(df, features):
    """
    Performs the following transformations on df:
        - selecting relevant features
        - scaling
        - adding noise
    :param df: dataframe as was read from the original csv.
    :param features: list of 2 features from the dataframe
    :return: transformed data as numpy array of shape (n, 2)
    """
    df = df[features]
    df.apply(sum_scaling)
    return np.array(df)


def choose_initial_centroids(data, k):
    """
    :param data: dataset as numpy array of shape (n, 2)
    :param k: number of clusters
    :return: numpy array of k random items from dataset
    """
    n = data.shape[0]
    indices = np.random.choice(range(n), k, replace=False)
    return data[indices]


def kmeans(data, k):
    """
    Running kmeans clustering algorithm.
    :param data: numpy array of shape (n, 2)
    :param k: desired number of cluster
    :return:
    * labels - numpy array of size n, where each entry is the predicted label (cluster number)
    * centroids - numpy array of shape (k, 2), centroid for each cluster.
    """

    centroids = choose_initial_centroids(data, k)
    prev_centroids = centroids
    labels = assign_to_clusters(data, centroids)
    current_centroids = recompute_centroids(data, labels, k)

    while not np.array_equal(prev_centroids, current_centroids):
        prev_centroids = current_centroids
        labels = assign_to_clusters(data, current_centroids)
        current_centroids = recompute_centroids(data, labels, k)

    centroids = current_centroids
    return labels, centroids


def dist(x, y):
    """
    Euclidean distance between vectors x, y
    :param x: numpy array of size n
    :param y: numpy array of size n
    :return: the euclidean distance
    """
    distance = 0
    for values_x, values_y in zip(x, y):
        print(values_y, values_x)
        distance += pow(values_x - values_y, 2)

    distance = sqrt(distance)
    return distance


def build_distance_matrix(data, centroids):
    k = len(centroids)
    n = len(data)
    distance_matrix = np.zeros((n, k))
    for cent_index, centroid in enumerate(centroids):
        for point in range(n):
            distance_matrix[point][cent_index] = dist(data[point], centroid)
    return distance_matrix


def assign_to_clusters(data, centroids):
    """
    Assign each data point to a cluster based on current centroids
    :param data: data as numpy array of shape (n, 2)
    :param centroids: current centroids as numpy array of shape (k, 2)
    :return: numpy array of size n
    """

    k = len(centroids)
    n = len(data)
    distance_matrix = build_distance_matrix(data, centroids)
    labels = np.zeros(n)
    for point in range(n):
        labels[point] = np.where(distance_matrix[point] == min(distance_matrix[point]))[0]

    return labels


def recompute_centroids(data, labels, k):
    """
    Recomputes new centroids based on the current assignment
    :param data: data as numpy array of shape (n, 2)
    :param labels: current assignments to clusters for each data point, as numpy array of size n
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    centroids = np.zeros((k, 2))
    for index, label in enumerate(labels):
        label = int(label)
        centroids[label][0] += data[index][0]
        centroids[label][1] += data[index][1]

    for index in range(k):
        centroids[index][0] /= np.count_nonzero(labels == index)
        centroids[index][1] /= np.count_nonzero(labels == index)

    return centroids


