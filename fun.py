import numpy as np


def confusion_matrix(expected, predicted, labels):
    expected = np.array(expected)
    predicted = np.array(predicted)
    num_labels = len(labels)
    matrix = np.zeros((num_labels, num_labels), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for e, p in zip(expected, predicted):
        if e in label_to_index and p in label_to_index:
            matrix[label_to_index[e], label_to_index[p]] += 1
    return matrix


def evaluate_predictions(expected, predicted):
    if len(expected) != len(predicted):
        raise ValueError("Both lists must have the same length.")

    arr1 = np.array(expected)
    arr2 = np.array(predicted)

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
