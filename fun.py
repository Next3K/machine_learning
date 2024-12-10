from tfidf import calculate_value_between_docs


def calculate_total_distances(selected, vectors, mask):
    distances = []
    columns = vectors.columns
    for i in range(0, len(vectors)):
        total = 0
        for j in range(1, len(columns)):
            if mask[j - 1] == 0:
                continue
            value_a = selected.values[j - 1]
            value_b = vectors.iloc[i, j]
            if columns[j] == "overview":
                total += get_cos_distance(value_a, value_b)
            elif columns[j] == "production_companies" or columns[j] == "production_countries" or columns[j] == "genres":
                total += array_text_feature(value_a, value_b)
            else:
                total += euclidean_distance(value_a, value_b)
        distances.append(total)
    return distances


def euclidean_distance(a, b):
    return abs(a - b)


def array_text_feature(string_a, string_b):
    array_a = string_a.split(",")
    array_b = string_b.split(",")
    length = max(len(array_a), len(array_b))
    value = set(array_a).intersection(array_b)
    similarity = len(value) / length
    return 1 - similarity


def get_cos_distance(doc_1, doc_2):
    return 1 - calculate_value_between_docs(doc_1, doc_2)


def get_average(idx, dataset):
    sum = 0
    for i in range(0, len(idx)):
        sum += dataset.iloc[idx[i], 0]
    return int(round(sum / len(idx), 0))


def get_label(idx, dataset):
    labels_dict = {}
    labels = [dataset.iloc[idx[i], 0] for i in range(0, len(idx))]
    for label in labels:
        if label in labels_dict:
            labels_dict[label] += 1
        else:
            labels_dict[label] = 1
    common_labels = [label for label, count in labels_dict.items() if count == max(labels_dict.values())]
    return common_labels[0]


import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_predictions(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    arr1 = np.array(list1)
    arr2 = np.array(list2)

    accuracy = np.mean(arr1 == arr2) * 100
    accuracy_plus_minus = np.mean(np.abs(arr1 - arr2) <= 1) * 100
    average_abs_error = np.mean(np.abs(arr1 - arr2))
    matrix_error = confusion_matrix(arr1, arr2, labels=[0, 1, 2, 3, 4, 5])

    return {
        "accuracy": accuracy,
        "accuracy_plus_minus": accuracy_plus_minus,
        "average_abs_error": average_abs_error,
        "matrix_error": matrix_error
    }
