import numpy as np
import matplotlib.pyplot as plt

k = 5

#plotok:
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
    plt.imshow(distances.T, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Euclidean Distance')
    plt.title('Euclidean Distance from Centroids')
    plt.xlabel('Test Samples')
    plt.ylabel('Centroids')
    plt.show()

#tavolsagok:
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def cosine_similarity_custom(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0 
    return dot_product / (norm_vec1 * norm_vec2)

def knn(train_data, test_data, k, distance_metric):
    predictions = []
    for test_point in test_data:
        distances = []
        for train_point in train_data:
            if distance_metric == 'euclidean':
                dist = euclidean_distance(test_point[:-1], train_point[:-1])
            elif distance_metric == 'cosine':
                dist = 1 - cosine_similarity_custom(test_point[:-1], train_point[:-1])
            distances.append((train_point, dist))
        distances.sort(key=lambda x: x[1])
        k_nearest = distances[:k]
        k_nearest_labels = [point[0][-1] for point in k_nearest]
        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(prediction)
    return predictions

def centroid_c(train_data, test_data, distance_metric):
    centroids = {label: np.mean(train_data[train_data[:, -1] == label][:, :-1], axis=0) 
                 for label in np.unique(train_data[:, -1])}
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
    train_data = np.loadtxt('optdigits.tra', delimiter=',')
    test_data = np.loadtxt('optdigits.tes', delimiter=',')

    knn_euclidean_predictions = knn(train_data, test_data, k, 'euclidean')
    accuracy_knn_euclidean = np.mean(knn_euclidean_predictions == test_data[:, -1])
    knn_train_predictions = knn(train_data, train_data, k, 'euclidean')
    accuracy_knn_train = np.mean(knn_train_predictions == train_data[:, -1])
    print("kNN: Euclidean distance - Train Accuracy:", accuracy_knn_train)
    print("kNN: Euclidean distance - Test Accuracy:", accuracy_knn_euclidean)

    knn_cosine_predictions = knn(train_data, test_data, k, 'cosine')
    accuracy_knn_cosine = np.mean(knn_cosine_predictions == test_data[:, -1])
    knn_train_predictions = knn(train_data, train_data, k, 'cosine')
    accuracy_knn_train = np.mean(knn_train_predictions == train_data[:, -1])
    print("kNN: Cosine similarity - Train Accuracy:", accuracy_knn_train)
    print("kNN: Cosine similarity - Test Accuracy:", accuracy_knn_cosine)

    centroid_euclidean_predictions = centroid_c(train_data, test_data, 'euclidean')
    accuracy_centroid_euclidean = np.mean(centroid_euclidean_predictions == test_data[:, -1])
    centroid_train_predictions = centroid_c(train_data, train_data, 'euclidean')
    accuracy_centroid_train = np.mean(centroid_train_predictions == train_data[:, -1])
    print("Centroid: Euclidean distance - Train Accuracy:", accuracy_centroid_train)
    print("Centroid: Euclidean distance - Test Accuracy:", accuracy_centroid_euclidean)

    centroid_cosine_predictions = centroid_c(train_data, test_data, 'cosine')
    accuracy_centroid_cosine = np.mean(centroid_cosine_predictions == test_data[:, -1])
    centroid_train_predictions = centroid_c(train_data, train_data, 'cosine')
    accuracy_centroid_train = np.mean(centroid_train_predictions == train_data[:, -1])
    print("Centroid: Cosine similarity - Train Accuracy:", accuracy_centroid_train)
    print("Centroid: Cosine similarity - Test Accuracy:", accuracy_centroid_cosine)

    centroids = [np.mean(train_data[train_data[:, -1] == label][:, :-1], axis=0) 
                 for label in np.unique(train_data[:, -1])]
    plot_centroids(centroids)


    distances = np.zeros((len(centroids), len(test_data)))
    for i, centroid in enumerate(centroids):
        for j, test_point in enumerate(test_data):
            distances[i, j] = np.linalg.norm(test_point[:-1] - centroid)

    plot_distance_heatmap(distances)

if __name__ == "__main__":
    main()
