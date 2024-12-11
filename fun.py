import numpy as np
from sklearn.metrics import confusion_matrix


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
