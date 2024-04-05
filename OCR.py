#Korpos Botond
#kbim2251

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict


k = 5

#beolvasasok es kimenetek:
def read_data():
    train_data = np.loadtxt('optdigits.tra', delimiter=',')
    test_data = np.loadtxt('optdigits.tes', delimiter=',')

    return train_data, test_data

def percentage_counter(grouped_points, predictions):
    prediction_dict = defaultdict(int)
    for prediction in predictions:
        prediction_dict[prediction] += 1

    for label in sorted(grouped_points.keys()):
        perc = abs(grouped_points[label] - prediction_dict[label])
        if(perc != 0):
            print(str(int(label)) + ": " + str(100 - ((perc * 100) / grouped_points[label])))
        else:
            print(str(int(label)) + ": " + str(100))

    print()

def print_results(train_data, test_data, centroids):
    
    #csoportositas elvart cimkek szerint
    grouped_points = defaultdict(int)
    last_indices = test_data[:, -1]

    for i, last_index in enumerate(last_indices):
        grouped_points[last_index] += 1

    grouped_points_train = defaultdict(int)
    last_indices = train_data[:, -1]

    for i, last_index in enumerate(last_indices):
        grouped_points_train[last_index] += 1


    knn_euclidean_predictions = knn(train_data, test_data, k, 'euclidean')
    accuracy_knn_euclidean = np.mean(knn_euclidean_predictions == test_data[:, -1])
    knn_train_predictions = knn(train_data, train_data, k, 'euclidean')
    accuracy_knn_train = np.mean(knn_train_predictions == train_data[:, -1])
    print("kNN: Euclidean distance - Train Accuracy:", accuracy_knn_train)
    print("kNN: Euclidean distance - Test Accuracy:", accuracy_knn_euclidean)       

    knn_cosine_predictions = knn(train_data, test_data, k, 'cosine')
    accuracy_knn_cosine = np.mean(knn_cosine_predictions == test_data[:, -1])
    knn_train_predictions_cos = knn(train_data, train_data, k, 'cosine')
    accuracy_knn_train = np.mean(knn_train_predictions_cos == train_data[:, -1])
    print("kNN: Cosine similarity - Train Accuracy:", accuracy_knn_train)
    print("kNN: Cosine similarity - Test Accuracy:", accuracy_knn_cosine)
    
    centroid_euclidean_predictions = centroid_c(test_data, 'euclidean', centroids)
    accuracy_centroid_euclidean = np.mean(centroid_euclidean_predictions == test_data[:, -1])
    centroid_train_predictions = centroid_c(train_data, 'euclidean', centroids)
    accuracy_centroid_train = np.mean(centroid_train_predictions == train_data[:, -1])
    print("Centroid: Euclidean distance - Train Accuracy:", accuracy_centroid_train)
    print("Centroid: Euclidean distance - Test Accuracy:", accuracy_centroid_euclidean)

    centroid_cosine_predictions = centroid_c(test_data, 'cosine', centroids)
    accuracy_centroid_cosine = np.mean(centroid_cosine_predictions == test_data[:, -1])
    centroid_train_predictions_cos = centroid_c(train_data, 'cosine', centroids)
    accuracy_centroid_train = np.mean(centroid_train_predictions_cos == train_data[:, -1])
    print("Centroid: Cosine similarity - Train Accuracy:", accuracy_centroid_train)
    print("Centroid: Cosine similarity - Test Accuracy:", accuracy_centroid_cosine)


    print("Test:")
    print("Knn:")
    percentage_counter(grouped_points, knn_euclidean_predictions)
    percentage_counter(grouped_points, knn_cosine_predictions)
    print("Centroid:")
    percentage_counter(grouped_points, centroid_euclidean_predictions)
    percentage_counter(grouped_points, centroid_cosine_predictions)

    print("Train:")
    print("Knn:")
    percentage_counter(grouped_points_train, knn_train_predictions)
    percentage_counter(grouped_points_train, knn_train_predictions_cos)
    print("Centroid:")
    percentage_counter(grouped_points_train, centroid_train_predictions)
    percentage_counter(grouped_points_train, centroid_train_predictions_cos)

#plotok:
def plotting(train_data, test_data, centroids): 
    plot_centroids(centroids)  

    #distance heatmap:
    grouped_points = defaultdict(list)
    last_indices = test_data[:, -1]

    # Group points by their labels
    for last_index in last_indices:
        grouped_points[int(last_index)].append(last_index)

    distances = np.zeros((len(test_data), len(centroids)))  # Initialize distances array

    l = 0
    for label, points in grouped_points.items():
        for test_point in points:
            for k, centroid in enumerate(centroids):
                distances[l, k] = euclidean_distance(test_point, centroid)
            l += 1

    plot_distance_heatmap(distances)
    #-----------------------------------------------------------------------#
    #cosine:
    '''
    cos_distance = np.zeros((len(test_data), len(test_data)))
    
    i1 = 0
    i2 = 0
    for i in range(len(grouped_points)):
        for j in range(len(grouped_points)):
            for ii in range(len(grouped_points[i])):
                for jj in range(len(grouped_points[j])):
                    cos_distance[i1][i2] = cosine_similarity_custom(test_data[ii][:-1], test_data[jj][:-1])
                    i2 += 1
                i2 = 0
                i1 += 1

    plot_distance_cos_heatmap(cos_distance)
    '''


def plot_digit(digit):
    plt.imshow(digit.reshape(8, 8), cmap='gray')
    plt.axis('off')
    plt.show()

def plot_centroids(centroids):
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(2):
        for j in range(5):
            axs[i, j].imshow(centroids[i*5 + j].reshape(8, 8), cmap='gray')
            axs[i, j].axis('off')
    plt.show()

def plot_distance_heatmap(distances):
    plt.imshow(distances, cmap='hot_r', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Euclidean Distance')
    plt.title('Euclidean Distance from Centroids')
    plt.xlabel('Centroids')
    plt.ylabel('Test Samples')
    plt.show()

def plot_distance_cos_heatmap(distances):
    plt.imshow(distances, cmap='hot_r', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('Cosine similarity for all test cases')
    plt.xlabel('Test Samples')
    plt.ylabel('Test Samples')
    plt.show()

#tavolsagok:
#sajat fuggvenyek (numpy nelkul):
"""
def euclidean_distance(vec1, vec2):
    squared_diff_sum = 0.0
    for i in range(len(vec1)):
        squared_diff_sum += (vec1[i] - vec2[i]) ** 2
    
    return math.sqrt(squared_diff_sum)

def dot_product(vec1, vec2):
    return sum(x * y for x, y in zip(vec1, vec2))

def magnitude(vec):
    return math.sqrt(sum(x ** 2 for x in vec))

def cosine_similarity_custom(vec1, vec2): 
    dot_prod = dot_product(vec1, vec2)
    mag1 = magnitude(vec1)
    mag2 = magnitude(vec2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    return dot_prod / (mag1 * mag2)
"""

#numpy hasznalva, ezzel sokkal gyorsabb
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_similarity_custom(point1, point2):
    dot_product = np.dot(point1, point2)
    norm_product = np.linalg.norm(point1) * np.linalg.norm(point2)
    if(dot_product == 0 or norm_product == 0):
        return 0
    return dot_product / norm_product

#centroid szamolo:
def calculate_centroids(train_data):
    centroids = {label: np.mean(train_data[train_data[:, -1] == label][:, :-1], axis=0) 
                 for label in np.unique(train_data[:, -1])}
    
    return centroids

#knn es centroidos megoldas:
def knn(train_data, test_data, k, distance_metric):
    predictions = []
    train_points = np.array([point[:-1] for point in train_data])
    train_labels = np.array([point[-1] for point in train_data], dtype=int)
    test_points = np.array([point[:-1] for point in test_data])

    for test_point in test_points:
        if distance_metric == 'euclidean':
            #idk miert de igy nagyon gyors :)
            distances = np.sqrt(np.sum((train_points - test_point) ** 2, axis=1))
        elif distance_metric == 'cosine':
            cos_sim = np.dot(train_points, test_point) / (np.linalg.norm(train_points, axis=1) * np.linalg.norm(test_point))
            distances = 1 - cos_sim
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        prediction = np.argmax(np.bincount(nearest_labels))
        predictions.append(prediction)

    return predictions

def centroid_c(test_data, distance_metric, centroids):
    predictions = []
    for test_point in test_data:
        distances = {}
        for label, centroid in centroids.items():
            if distance_metric == 'euclidean':
                distances[label] = euclidean_distance(test_point[:-1], centroid)
            elif distance_metric == 'cosine':
                distances[label] = 1 - cosine_similarity_custom(test_point[:-1], centroid)
        prediction = min(distances, key=distances.get)
        predictions.append(prediction)

    return predictions

def main():
    train_data, test_data = read_data()
    
    centroids = calculate_centroids(train_data)
    #print_results(train_data, test_data, centroids)
    plotting(train_data, test_data, centroids)

    return 0

if __name__ == '__main__':
    main()
